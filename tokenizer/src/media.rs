//! Media inputs of a request: extraction from the protocol body, digests,
//! pad values, planning and expansion into the token list.
//!
//! Images and videos are content blocks of the client request. Besides the
//! protocol's own fields, a block may carry three ModelWeave extensions:
//!
//! ```json
//! {"type": "image_url", "image_url": {"url": "https://..."},
//!  "metadata": {"digest": "<sha256 hex>", "width": 1920, "height": 1080},
//!  "pixel_control_options": {"max_long_side": 1344, "max_total_pixels": 4000000}}
//! {"type": "video_url", "video_url": {"url": "https://..."},
//!  "metadata": {"digest": "...", "width": 1280, "height": 720, "duration_ms": 10000,
//!               "total_frames": 300, "video_fps": 30},
//!  "frame_extract_options": {"fps": 2, "pixel_control_options": {"max_long_side": 672}}}
//! ```
//!
//! Before protocol conversion, every media block is replaced by an image
//! block with a synthetic URL that numbers it, so rendered placeholders map
//! back to the media in order.

use serde::Deserialize;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

#[cfg(test)]
use crate::protocol::Span;
use crate::protocol::{
    FrameExtractOptions, MediaItem, MediaMetadata, MediaPlan, MediaSource,
    PROTOCOL_CHAT_COMPLETIONS, PROTOCOL_MESSAGES, PROTOCOL_RESPONSES, PixelControlOptions,
    Placement,
};
use crate::vision::{VideoMeta, VisionSpec};

/// pad_value base: values live in [2^30, 2^31), far outside any vocabulary.
pub const PAD_SHIFT: u32 = 1 << 30;

const SYNTHETIC_PREFIX: &str = "http://modelweave.invalid/media/";

/// The pad value of a media reference: `2^30 + (sha256(reference)[0..8] as
/// big-endian u64 & (2^30 - 1))`. Equal media share a pad value, so the
/// token list of a multimodal prompt is a stable prefix-cache key (the
/// SGLang approach); engines locate items by their placements.
pub fn pad_value(reference: &str) -> u32 {
    let h = Sha256::digest(reference.as_bytes());
    let mut first = [0u8; 8];
    first.copy_from_slice(&h[..8]);
    PAD_SHIFT + (u64::from_be_bytes(first) & ((1 << 30) - 1)) as u32
}

/// The reference of kept frame `k` of a video.
pub fn kept_frame_reference(digest: &str, k: usize) -> String {
    format!("{digest}_kf_{k:04}")
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let h = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in h {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Language-neutral execution-plan fingerprint (internal/mwip/media.go).
/// Excludes source URLs and absolute prompt positions so cached embeddings
/// remain reusable when a signed URL rotates or the conversation grows.
pub fn plan_id(modality: &str, digest: &str, plan: &MediaPlan) -> String {
    let mut bytes = b"mwip-media-plan-v2\0".to_vec();
    fn u64_le(bytes: &mut Vec<u8>, n: u64) {
        bytes.extend_from_slice(&n.to_le_bytes());
    }
    fn string(bytes: &mut Vec<u8>, s: &str) {
        u64_le(bytes, s.len() as u64);
        bytes.extend_from_slice(s.as_bytes());
    }
    string(&mut bytes, modality);
    string(&mut bytes, digest);
    string(&mut bytes, plan.processor);
    for n in [
        plan.width as u64,
        plan.height as u64,
        plan.tokens as u64,
        plan.nframes,
        plan.patch_size as u64,
        plan.spatial_merge_size as u64,
        plan.temporal_patch_size as u64,
    ] {
        u64_le(&mut bytes, n);
    }
    u64_le(&mut bytes, plan.frame_indices.len() as u64);
    for n in &plan.frame_indices {
        u64_le(&mut bytes, *n);
    }
    u64_le(&mut bytes, plan.timestamps.len() as u64);
    for n in &plan.timestamps {
        u64_le(&mut bytes, n.to_bits());
    }
    sha256_hex(&bytes)
}

fn is_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Image,
    Video,
}

impl Modality {
    pub fn name(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
        }
    }
}

/// One image or video of a request, as extracted from the body.
#[derive(Debug, Clone)]
pub struct MediaInput {
    pub modality: Modality,
    /// `url` or `base64`.
    pub kind: &'static str,
    pub url: String,
    pub mime: String,
    /// Decoded bytes of an inline (data URL) source.
    pub bytes: Option<Vec<u8>>,
    pub detail: Option<String>,
    pub metadata: MediaMetadata,
    pub pixel_control: Option<PixelControlOptions>,
    pub frame_extract: Option<FrameExtractOptions>,
    /// Request location for error messages, e.g. `messages.0.content.1`.
    pub path: String,
}

impl MediaInput {
    /// The media digest: the caller's, or the sha256 of inline bytes.
    pub fn digest(&self) -> Result<String, String> {
        match (&self.metadata.digest, &self.bytes) {
            (Some(d), _) if is_digest(d) => Ok(d.clone()),
            (Some(d), _) => Err(format!(
                "{}: metadata.digest must be 64 lowercase hex characters (sha256), got {d:?}",
                self.path
            )),
            (None, Some(b)) => Ok(sha256_hex(b)),
            (None, None) => Err(format!(
                "{}: metadata.digest (sha256 of the {} bytes) is required for remote media",
                self.path,
                self.modality.name()
            )),
        }
    }

    /// Width and height: the caller's, or read from inline image bytes.
    pub fn dimensions(&self) -> Result<(u32, u32), String> {
        if let (Some(w), Some(h)) = (self.metadata.width, self.metadata.height) {
            return Ok((w, h));
        }
        if self.modality == Modality::Image
            && let Some(b) = &self.bytes
            && let Some(d) = probe_dimensions(b)
        {
            return Ok(d);
        }
        Err(format!(
            "{}: metadata.width and metadata.height are required{}",
            self.path,
            if self.modality == Modality::Image {
                " for remote images"
            } else {
                ""
            }
        ))
    }
}

#[derive(Deserialize)]
struct Extensions {
    #[serde(default)]
    metadata: Option<MediaMetadata>,
    #[serde(default)]
    pixel_control_options: Option<PixelControlOptions>,
    #[serde(default)]
    frame_extract_options: Option<FrameExtractOptions>,
}

/// Extract the media of a request body, replacing each media block with a
/// numbered synthetic image block.
pub fn extract(protocol: &str, body: &mut Value) -> Result<Vec<MediaInput>, String> {
    let mut out = Vec::new();
    match protocol {
        PROTOCOL_CHAT_COMPLETIONS | PROTOCOL_MESSAGES => {
            if let Some(msgs) = body.get_mut("messages").and_then(Value::as_array_mut) {
                for (i, m) in msgs.iter_mut().enumerate() {
                    if let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut) {
                        walk(protocol, blocks, &format!("messages.{i}.content"), &mut out)?;
                    }
                }
            }
        }
        PROTOCOL_RESPONSES => {
            if let Some(items) = body.get_mut("input").and_then(Value::as_array_mut) {
                for (i, item) in items.iter_mut().enumerate() {
                    for key in ["content", "output"] {
                        if let Some(blocks) = item.get_mut(key).and_then(Value::as_array_mut) {
                            walk(protocol, blocks, &format!("input.{i}.{key}"), &mut out)?;
                        }
                    }
                }
            }
        }
        _ => {}
    }
    Ok(out)
}

fn walk(
    protocol: &str,
    blocks: &mut [Value],
    path: &str,
    out: &mut Vec<MediaInput>,
) -> Result<(), String> {
    for (j, block) in blocks.iter_mut().enumerate() {
        let here = format!("{path}.{j}");
        let Some(obj) = block.as_object_mut() else {
            continue;
        };
        let typ = obj
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let found = match (protocol, typ.as_str()) {
            (PROTOCOL_CHAT_COMPLETIONS, "image_url") => {
                url_field(obj.get("image_url")).map(|(u, d)| (Modality::Image, u, d))
            }
            (PROTOCOL_CHAT_COMPLETIONS, "video_url") => {
                url_field(obj.get("video_url")).map(|(u, _)| (Modality::Video, u, None))
            }
            (PROTOCOL_CHAT_COMPLETIONS, "file") => obj
                .get("file_data")
                .and_then(Value::as_str)
                .map(|d| (Modality::Image, d.to_string(), None)),
            (PROTOCOL_MESSAGES, "image") => {
                anthropic_source(obj.get("source")).map(|u| (Modality::Image, u, None))
            }
            (PROTOCOL_MESSAGES, "video") => {
                anthropic_source(obj.get("source")).map(|u| (Modality::Video, u, None))
            }
            (PROTOCOL_MESSAGES, "tool_result") => {
                if let Some(inner) = obj.get_mut("content").and_then(Value::as_array_mut) {
                    walk(protocol, inner, &format!("{here}.content"), out)?;
                }
                None
            }
            (PROTOCOL_RESPONSES, "input_image") => {
                obj.get("image_url").and_then(Value::as_str).map(|u| {
                    let detail = obj
                        .get("detail")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    (Modality::Image, u.to_string(), detail)
                })
            }
            (PROTOCOL_RESPONSES, "input_video") => obj
                .get("video_url")
                .and_then(Value::as_str)
                .map(|u| (Modality::Video, u.to_string(), None)),
            _ => None,
        };
        let Some((modality, url, detail)) = found else {
            continue;
        };
        let ext: Extensions = serde_json::from_value(Value::Object(obj.clone()))
            .map_err(|e| format!("{here}: invalid media options: {e}"))?;
        if modality == Modality::Image && ext.frame_extract_options.is_some() {
            return Err(format!(
                "{here}: frame_extract_options apply to videos only"
            ));
        }
        if modality == Modality::Video && ext.pixel_control_options.is_some() {
            return Err(format!(
                "{here}: a video takes its pixel controls in frame_extract_options.pixel_control_options"
            ));
        }
        let (kind, mime, bytes) = if url.starts_with("data:") {
            let (mime, bytes) =
                decode_data_url(&url).ok_or_else(|| format!("{here}: invalid data URL"))?;
            ("base64", mime, Some(bytes))
        } else if url.len() >= 4 && url[..4].eq_ignore_ascii_case("http") {
            ("url", String::new(), None)
        } else {
            return Err(format!(
                "{here}: media URL must be an http(s) URL or a data URL"
            ));
        };
        let index = out.len();
        out.push(MediaInput {
            modality,
            kind,
            url,
            mime,
            bytes,
            detail,
            metadata: ext.metadata.unwrap_or_default(),
            pixel_control: ext.pixel_control_options,
            frame_extract: ext.frame_extract_options,
            path: here,
        });
        *block = synthetic_block(protocol, index);
    }
    Ok(())
}

/// `{"url": ..., "detail": ...}` or a bare URL string.
fn url_field(v: Option<&Value>) -> Option<(String, Option<String>)> {
    match v? {
        Value::String(s) => Some((s.clone(), None)),
        Value::Object(o) => {
            let url = o.get("url")?.as_str()?.to_string();
            let detail = o.get("detail").and_then(Value::as_str).map(str::to_string);
            Some((url, detail))
        }
        _ => None,
    }
}

/// Anthropic `source`: `{"type": "url", "url"}` or `{"type": "base64",
/// "media_type", "data"}` (as a data URL).
fn anthropic_source(v: Option<&Value>) -> Option<String> {
    let o = v?.as_object()?;
    match o.get("type")?.as_str()? {
        "url" => Some(o.get("url")?.as_str()?.to_string()),
        "base64" => Some(format!(
            "data:{};base64,{}",
            o.get("media_type")?.as_str()?,
            o.get("data")?.as_str()?
        )),
        _ => None,
    }
}

fn synthetic_block(protocol: &str, index: usize) -> Value {
    let url = format!("{SYNTHETIC_PREFIX}{index}");
    match protocol {
        PROTOCOL_MESSAGES => json!({"type": "image", "source": {"type": "url", "url": url}}),
        PROTOCOL_RESPONSES => json!({"type": "input_image", "image_url": url}),
        _ => {
            let mut image_url = Map::new();
            image_url.insert("url".into(), json!(url));
            json!({"type": "image_url", "image_url": image_url})
        }
    }
}

/// The media index of a rendered image source with a synthetic URL.
pub fn synthetic_index(url: &str) -> Option<usize> {
    url.strip_prefix(SYNTHETIC_PREFIX)?.parse().ok()
}

fn decode_data_url(data_url: &str) -> Option<(String, Vec<u8>)> {
    let parsed = data_url::DataUrl::process(data_url).ok()?;
    let mime = format!(
        "{}/{}",
        parsed.mime_type().type_,
        parsed.mime_type().subtype
    );
    let (bytes, _) = parsed.decode_to_vec().ok()?;
    Some((mime, bytes))
}

/// Width and height from an encoded image header.
pub fn probe_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

// ---------------------------------------------------------------------------
// Planning and expansion
// ---------------------------------------------------------------------------

/// A planned media item: its input and the token plan.
pub struct Planned<'a> {
    pub input: &'a MediaInput,
    pub digest: String,
    pub plan: MediaPlan,
}

/// Resolve the digest and dimensions of every media input and plan it.
pub fn plan_all<'a>(
    vision: &VisionSpec,
    inputs: &'a [MediaInput],
) -> Result<Vec<Planned<'a>>, String> {
    if inputs.len() > vision.max_media() {
        return Err(format!(
            "{} images and videos exceed the limit of {}",
            inputs.len(),
            vision.max_media()
        ));
    }
    inputs
        .iter()
        .map(|m| plan_one(vision, m, inputs.len()))
        .collect()
}

fn plan_one<'a>(
    vision: &VisionSpec,
    m: &'a MediaInput,
    media_count: usize,
) -> Result<Planned<'a>, String> {
    let digest = m.digest()?;
    let (width, height) = m.dimensions()?;
    let at = |e: String| format!("{}: {e}", m.path);
    let plan = match m.modality {
        Modality::Image => {
            let low = m.detail.as_deref() == Some("low");
            vision
                .plan_image(width, height, low, m.pixel_control.as_ref(), media_count)
                .map_err(at)?
        }
        Modality::Video => {
            let (Some(duration_ms), Some(video_fps)) =
                (m.metadata.duration_ms, m.metadata.video_fps)
            else {
                return Err(at(
                    "a video needs metadata.duration_ms and metadata.video_fps".into(),
                ));
            };
            let video = VideoMeta {
                width,
                height,
                duration_ms,
                total_frames: m.metadata.total_frames,
                video_fps,
            };
            vision
                .plan_video(&video, m.frame_extract.as_ref(), media_count)
                .map_err(at)?
        }
    };
    Ok(Planned {
        input: m,
        digest,
        plan,
    })
}

/// How a model lays media out in its token list.
pub struct MediaLayout {
    /// Collapsed form used only to compare the rendered prompt with HF golden data.
    #[cfg(test)]
    pub image_placeholder: Vec<u32>,
    #[cfg(test)]
    pub video_placeholder: Vec<u32>,
    /// Text before each kept video frame, from its timestamp in seconds.
    pub frame_text: fn(f64) -> String,
    /// Tokens around each kept frame's positions (Qwen-VL
    /// `<|vision_start|>` / `<|vision_end|>`).
    pub frame_open: Option<u32>,
    pub frame_close: Option<u32>,
}

/// A rendered placeholder: `len` tokens at `start` stand for media item
/// `media`.
pub struct Slot {
    pub start: usize,
    pub len: usize,
    pub media: usize,
}

/// Replace every placeholder with the positions its item takes in the engine
/// input, each holding the item's pad value. A video becomes its kept frames,
/// each preceded by its timestamp text (and wrapped in the layout's frame
/// tokens). `encode` tokenizes plain text.
pub fn expand(
    ids: &[u32],
    slots: &[Slot],
    planned: &[Planned],
    layout: &MediaLayout,
    encode: &dyn Fn(&str) -> Result<Vec<u32>, String>,
) -> Result<(Vec<u32>, Vec<MediaItem>), String> {
    if slots.len() != planned.len() {
        return Err(format!(
            "the prompt renders {} media placeholders for {} images and videos",
            slots.len(),
            planned.len()
        ));
    }
    let extra: usize = planned
        .iter()
        .map(|p| p.plan.tokens * p.plan.groups() + 8 * p.plan.timestamps.len())
        .sum();
    let mut out = Vec::with_capacity(ids.len() + extra);
    let mut items = Vec::with_capacity(planned.len());
    let mut pos = 0;
    for slot in slots {
        let p = planned
            .get(slot.media)
            .ok_or_else(|| format!("placeholder for unknown media item {}", slot.media))?;
        if slot.start < pos || slot.start + slot.len > ids.len() {
            return Err("media placeholders overlap".into());
        }
        out.extend_from_slice(&ids[pos..slot.start]);
        pos = slot.start + slot.len;
        #[cfg(test)]
        let start = out.len();
        let mut placements = Vec::new();
        let plan_id = plan_id(p.input.modality.name(), &p.digest, &p.plan);
        match p.input.modality {
            Modality::Image => {
                let pv = pad_value(&plan_id);
                placements.push(Placement {
                    offset: out.len(),
                    length: p.plan.tokens,
                    pad_value: pv,
                    reference: plan_id.clone(),
                    modality: "image",
                    timestamp: None,
                    embedding_offset: 0,
                });
                out.extend(std::iter::repeat_n(pv, p.plan.tokens));
            }
            Modality::Video => {
                for (k, ts) in p.plan.timestamps.iter().enumerate() {
                    out.extend(encode(&(layout.frame_text)(*ts))?);
                    out.extend(layout.frame_open);
                    let reference = kept_frame_reference(&plan_id, k);
                    let pv = pad_value(&reference);
                    placements.push(Placement {
                        offset: out.len(),
                        length: p.plan.tokens,
                        pad_value: pv,
                        reference,
                        modality: "frame",
                        timestamp: Some(*ts),
                        embedding_offset: k * p.plan.tokens,
                    });
                    out.extend(std::iter::repeat_n(pv, p.plan.tokens));
                    out.extend(layout.frame_close);
                }
            }
        }
        let m = p.input;
        items.push(MediaItem {
            index: items.len(),
            modality: m.modality.name(),
            digest: p.digest.clone(),
            source: MediaSource {
                kind: m.kind,
                url: m.url.clone(),
                mime: m.mime.clone(),
            },
            plan: p.plan.clone(),
            plan_id,
            #[cfg(test)]
            span: Span {
                offset: start,
                length: out.len() - start,
            },
            placements,
            #[cfg(test)]
            placeholder: match p.input.modality {
                Modality::Image => layout.image_placeholder.clone(),
                Modality::Video => layout.video_placeholder.clone(),
            },
        });
    }
    out.extend_from_slice(&ids[pos..]);
    Ok((out, items))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_values_reference() {
        // Reference values computed with Python's hashlib.sha256.
        let digest = sha256_hex(b"modelweave");
        assert_eq!(
            digest,
            "e7d0dd0ebf325e345772dee6b92f69e47d3640c03e981bd4e263cfd875d8c2c5"
        );
        assert_eq!(pad_value(&digest), 1_697_097_594);
        assert_eq!(pad_value(&kept_frame_reference(&digest, 0)), 1_123_785_688);
        assert_eq!(pad_value(&kept_frame_reference(&digest, 1)), 1_986_516_877);
        assert_eq!(
            kept_frame_reference(&digest, 7),
            format!("{digest}_kf_0007")
        );
    }

    #[test]
    fn extraction_replaces_media_blocks_in_every_protocol() {
        let d = sha256_hex(b"x");
        let mut chat = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "https://a/x.png", "detail": "low"},
             "metadata": {"digest": d, "width": 10, "height": 20}},
            {"type": "video_url", "video_url": {"url": "https://a/v.mp4"},
             "metadata": {"digest": d, "width": 64, "height": 48, "duration_ms": 1000, "video_fps": 10},
             "frame_extract_options": {"fps": 1}}
        ]}]});
        let media = extract(PROTOCOL_CHAT_COMPLETIONS, &mut chat).unwrap();
        assert_eq!(media.len(), 2);
        assert_eq!(
            (media[0].modality, media[0].detail.as_deref()),
            (Modality::Image, Some("low"))
        );
        assert_eq!(media[1].modality, Modality::Video);
        assert_eq!(media[1].frame_extract.as_ref().unwrap().fps, Some(1.0));
        assert_eq!(
            chat["messages"][0]["content"][2]["image_url"]["url"],
            format!("{SYNTHETIC_PREFIX}1")
        );
        assert!(chat["messages"][0]["content"][1].get("metadata").is_none());

        let mut messages = json!({"messages": [{"role": "user", "content": [
            {"type": "video", "source": {"type": "url", "url": "https://a/v.mp4"}},
            {"type": "tool_result", "tool_use_id": "t", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}
        ]}]});
        let media = extract(PROTOCOL_MESSAGES, &mut messages).unwrap();
        assert_eq!(media.len(), 2);
        assert_eq!(media[1].kind, "base64");
        assert_eq!(
            messages["messages"][0]["content"][0]["source"]["url"],
            format!("{SYNTHETIC_PREFIX}0")
        );

        let mut responses = json!({"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "https://a/x.png"},
            {"type": "input_video", "video_url": "https://a/v.mp4"}]}]});
        assert_eq!(
            extract(PROTOCOL_RESPONSES, &mut responses).unwrap().len(),
            2
        );
        assert_eq!(responses["input"][0]["content"][1]["type"], "input_image");
    }

    #[test]
    fn digests_and_dimensions() {
        let mut body = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://a/x.png"}}]}]});
        let m = extract(PROTOCOL_CHAT_COMPLETIONS, &mut body)
            .unwrap()
            .remove(0);
        assert!(m.digest().unwrap_err().contains("metadata.digest"));
        assert!(m.dimensions().unwrap_err().contains("metadata.width"));
        let mut bad = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://a/x.png"}, "metadata": {"digest": "abc"}}]}]});
        let m = extract(PROTOCOL_CHAT_COMPLETIONS, &mut bad)
            .unwrap()
            .remove(0);
        assert!(m.digest().unwrap_err().contains("64 lowercase hex"));
        let mut opts = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://a/x.png"}, "frame_extract_options": {"fps": 1}}]}]});
        assert!(
            extract(PROTOCOL_CHAT_COMPLETIONS, &mut opts)
                .unwrap_err()
                .contains("videos only")
        );
        let mut unknown = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://a/x.png"}, "pixel_control_options": {"max_pixels": 1}}]}]});
        assert!(
            extract(PROTOCOL_CHAT_COMPLETIONS, &mut unknown)
                .unwrap_err()
                .contains("invalid media options")
        );
    }
}
