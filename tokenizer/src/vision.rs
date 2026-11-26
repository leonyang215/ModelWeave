//! Token planning of images and videos: how many positions an item takes in
//! the engine input, computed from the metadata the caller sends.
//!
//! Two layers:
//!
//! * the engineering layer (pixel controls, shared by every model): a
//!   long-side limit applied to the source size, a pixel budget checked on
//!   the result, the aspect ratio and dimension limits;
//! * the model layer: the model's own resize, token count and frame
//!   sampling:
//!   * DeepSeek V4.1: deepseek-recipe's `ImageTokenSpec` (14-pixel patches,
//!     downsample 3, one newline token per row plus start and end tokens, at
//!     most 1024 tokens, small images upscaled to 544² pixels); videos are
//!     sampled with `smart_nframes` (qwen-vl-utils) and every kept frame is
//!     planned like an image;
//!   * Qwen-VL (Qwen2.5-VL, Qwen3-VL): `smart_resize` to multiples of
//!     patch × merge within the processor's pixel range, one token per
//!     merged patch; videos are sampled like transformers' Qwen3-VL video
//!     processor, two frames form one temporal patch, and the per-frame pixel
//!     budget follows the reference implementation (qwen-vl-utils).
//!
//! Encoders execute the resulting plan, not a second copy of these rules.

use deepseek_recipe_core::multimodal::ImageTokenSpec;
use serde_json::Value;

use crate::protocol::{FrameExtractOptions, MediaPlan, PixelControlOptions};

/// Engineering limits that apply to every model.
#[derive(Debug, Clone)]
pub struct MediaLimits {
    /// Long-side limit of `detail: "low"` images.
    pub low_detail_max_long_side: u32,
    /// Largest accepted source width or height, and the limit from
    /// `many_media_threshold` items on.
    pub max_dimension: u32,
    pub max_dimension_many: u32,
    pub many_media_threshold: usize,
    /// Most images and videos per request.
    pub max_media: usize,
    /// Largest accepted long-side / short-side ratio.
    pub max_aspect_ratio: f64,
    /// Pixel budgets when pixel controls are given without one.
    pub image_default_max_total_pixels: u64,
    pub video_default_max_total_pixels: u64,
}

impl Default for MediaLimits {
    fn default() -> Self {
        Self {
            low_detail_max_long_side: 512,
            max_dimension: 8192,
            max_dimension_many: 4096,
            many_media_threshold: 15,
            max_media: 600,
            max_aspect_ratio: 200.0,
            image_default_max_total_pixels: 3584 * 3584,
            video_default_max_total_pixels: 301_056_000,
        }
    }
}

impl MediaLimits {
    fn check_source(&self, width: u32, height: u32, media_count: usize) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Err(format!("invalid size {width}x{height}"));
        }
        let limit = if media_count >= self.many_media_threshold {
            self.max_dimension_many
        } else {
            self.max_dimension
        };
        if width > limit || height > limit {
            return Err(format!(
                "{width}x{height} exceeds the {limit}-pixel dimension limit"
            ));
        }
        let ratio = width.max(height) as f64 / width.min(height) as f64;
        if ratio > self.max_aspect_ratio {
            return Err(format!(
                "aspect ratio {ratio:.1} exceeds {}",
                self.max_aspect_ratio
            ));
        }
        Ok(())
    }

    fn check_budget(
        &self,
        what: &str,
        pixels: Option<&PixelControlOptions>,
        default: u64,
        total: u64,
    ) -> Result<(), String> {
        let Some(p) = pixels else {
            return Ok(());
        };
        let budget = p.max_total_pixels.unwrap_or(default);
        if total > budget {
            return Err(format!(
                "{what} = {total} pixels exceeds max_total_pixels {budget}"
            ));
        }
        Ok(())
    }
}

/// DeepSeek V4.1 vision settings.
#[derive(Debug, Clone)]
pub struct DeepseekVision {
    pub image: ImageTokenSpec,
    /// Frames fused into one kept frame by the vision encoder (1: none).
    pub video_temporal_patch_size: u32,
    /// Sampling rate without `frame_extract_options.fps`.
    pub video_default_fps: f64,
    /// Long-side limit of video frames without a pixel control.
    pub video_frame_max_long_side: u32,
}

/// Qwen-VL vision settings (the image and video processor configuration).
#[derive(Debug, Clone, PartialEq)]
pub struct QwenVision {
    pub patch_size: u32,
    pub merge_size: u32,
    pub temporal_patch_size: u32,
    pub image_min_pixels: u64,
    pub image_max_pixels: u64,
    pub video_min_pixels: u64,
    pub video_max_pixels: u64,
    /// Sampling rate without `frame_extract_options.fps`.
    pub video_fps: f64,
    pub video_min_frames: u32,
    pub video_max_frames: u32,
    /// Per-frame token ceiling of the reference implementation
    /// (qwen-vl-utils `VIDEO_MAX_TOKEN_NUM`); `None` spends the whole video
    /// pixel budget on the sampled frames.
    pub video_max_frame_tokens: Option<u32>,
}

impl QwenVision {
    /// Qwen3-VL processor defaults (preprocessor_config.json and
    /// video_preprocessor_config.json of the Qwen3-VL releases).
    pub fn qwen3_vl() -> Self {
        Self {
            patch_size: 16,
            merge_size: 2,
            temporal_patch_size: 2,
            image_min_pixels: 65_536,
            image_max_pixels: 16_777_216,
            video_min_pixels: 4_096,
            video_max_pixels: 25_165_824,
            video_fps: 2.0,
            video_min_frames: 4,
            video_max_frames: 768,
            video_max_frame_tokens: Some(768),
        }
    }

    /// Override the defaults with a Hugging Face processor configuration
    /// (preprocessor_config.json or video_preprocessor_config.json).
    pub fn apply_processor_config(&mut self, cfg: &Value, video: bool) {
        let get = |k: &str| cfg.get(k).and_then(Value::as_u64);
        if let Some(v) = get("patch_size") {
            self.patch_size = v as u32;
        }
        if let Some(v) = get("merge_size") {
            self.merge_size = v as u32;
        }
        if let Some(v) = get("temporal_patch_size") {
            self.temporal_patch_size = v as u32;
        }
        let size = |k: &str| {
            cfg.get("size")
                .and_then(|s| s.get(k))
                .and_then(Value::as_u64)
        };
        let min = size("shortest_edge").or_else(|| get("min_pixels"));
        let max = size("longest_edge").or_else(|| get("max_pixels"));
        if video {
            if let Some(v) = min {
                self.video_min_pixels = v;
            }
            if let Some(v) = max {
                self.video_max_pixels = v;
            }
            if let Some(v) = cfg.get("fps").and_then(Value::as_f64) {
                self.video_fps = v;
            }
            if let Some(v) = get("min_frames") {
                self.video_min_frames = v as u32;
            }
            if let Some(v) = get("max_frames") {
                self.video_max_frames = v as u32;
            }
        } else {
            if let Some(v) = min {
                self.image_min_pixels = v;
            }
            if let Some(v) = max {
                self.image_max_pixels = v;
            }
        }
    }

    fn factor(&self) -> u32 {
        self.patch_size * self.merge_size
    }

    fn tokens(&self, width: u32, height: u32) -> usize {
        ((height / self.patch_size) * (width / self.patch_size)
            / (self.merge_size * self.merge_size)) as usize
    }
}

/// The model layer of a vision model.
#[derive(Debug, Clone)]
pub enum Planner {
    Deepseek(DeepseekVision),
    Qwen(QwenVision),
}

/// Vision settings of one model.
#[derive(Debug, Clone)]
pub struct VisionSpec {
    pub limits: MediaLimits,
    pub planner: Planner,
}

impl VisionSpec {
    pub fn deepseek_v41() -> Self {
        Self {
            limits: MediaLimits::default(),
            planner: Planner::Deepseek(DeepseekVision {
                image: ImageTokenSpec::v41(),
                video_temporal_patch_size: 1,
                video_default_fps: 2.0,
                video_frame_max_long_side: 672,
            }),
        }
    }

    pub fn qwen(vision: QwenVision) -> Self {
        Self {
            limits: MediaLimits::default(),
            planner: Planner::Qwen(vision),
        }
    }

    pub fn max_media(&self) -> usize {
        self.limits.max_media
    }

    /// Plan one image.
    pub fn plan_image(
        &self,
        width: u32,
        height: u32,
        low_detail: bool,
        pixels: Option<&PixelControlOptions>,
        media_count: usize,
    ) -> Result<MediaPlan, String> {
        let l = &self.limits;
        l.check_source(width, height, media_count)?;
        let mut limit = pixels.and_then(|p| p.max_long_side);
        if low_detail {
            limit = Some(limit.map_or(l.low_detail_max_long_side, |x| {
                x.min(l.low_detail_max_long_side)
            }));
        }
        let (sw, sh) = match limit {
            Some(limit) => limit_long_side(width, height, limit),
            None => (width, height),
        };
        let (w, h, tokens) = match &self.planner {
            Planner::Deepseek(d) => {
                let r = d
                    .image
                    .calc_resize(sw as usize, sh as usize)
                    .map_err(|e| e.to_string())?;
                (r.best_width as u32, r.best_height as u32, r.token_count)
            }
            Planner::Qwen(q) => {
                let (h, w) =
                    image_smart_resize(sh, sw, q.factor(), q.image_min_pixels, q.image_max_pixels)?;
                (w, h, q.tokens(w, h))
            }
        };
        l.check_budget(
            &format!("image resized to {w}x{h}"),
            pixels,
            l.image_default_max_total_pixels,
            w as u64 * h as u64,
        )?;
        Ok(MediaPlan {
            processor: match &self.planner {
                Planner::Deepseek(_) => "deepseek_v4_1.v1",
                Planner::Qwen(_) => "qwen_vl.v1",
            },
            width: w,
            height: h,
            tokens,
            nframes: 1,
            patch_size: match &self.planner {
                Planner::Deepseek(_) => 14,
                Planner::Qwen(q) => q.patch_size,
            },
            spatial_merge_size: match &self.planner {
                Planner::Deepseek(_) => 3,
                Planner::Qwen(q) => q.merge_size,
            },
            temporal_patch_size: match &self.planner {
                Planner::Deepseek(d) => d.video_temporal_patch_size.max(1),
                Planner::Qwen(q) => q.temporal_patch_size.max(1),
            },
            frame_indices: Vec::new(),
            timestamps: Vec::new(),
        })
    }

    /// Plan one video: sample frames, then size the frames.
    pub fn plan_video(
        &self,
        video: &VideoMeta,
        opts: Option<&FrameExtractOptions>,
        media_count: usize,
    ) -> Result<MediaPlan, String> {
        let l = &self.limits;
        l.check_source(video.width, video.height, media_count)?;
        let fps = opts.and_then(|o| o.fps);
        let fixed = opts.and_then(|o| o.nframes);
        if fps.is_some() && fixed.is_some() {
            return Err("frame_extract_options.fps and nframes are mutually exclusive".into());
        }
        if let Some(f) = fps
            && (f.is_nan() || f <= 0.0)
        {
            return Err(format!("frame_extract_options.fps must be > 0, got {f}"));
        }
        if video.video_fps.is_nan() || video.video_fps <= 0.0 {
            return Err(format!(
                "metadata.video_fps must be > 0, got {}",
                video.video_fps
            ));
        }
        let pixels = opts.and_then(|o| o.pixel_control_options.as_ref());
        let plan = match &self.planner {
            Planner::Deepseek(d) => plan_video_deepseek(d, video, fps, fixed, pixels)?,
            Planner::Qwen(q) => plan_video_qwen(q, video, fps, fixed, pixels)?,
        };
        let kept = plan.groups() as u64;
        l.check_budget(
            &format!("video frames {}x{} x {kept}", plan.width, plan.height),
            pixels,
            l.video_default_max_total_pixels,
            plan.width as u64 * plan.height as u64 * kept,
        )?;
        Ok(plan)
    }
}

fn plan_video_deepseek(
    d: &DeepseekVision,
    video: &VideoMeta,
    fps: Option<f64>,
    fixed: Option<u32>,
    pixels: Option<&PixelControlOptions>,
) -> Result<MediaPlan, String> {
    let tps = d.video_temporal_patch_size.max(1);
    let nframes = smart_nframes(video, fps, fixed, tps as u64, d.video_default_fps)?;
    let kept = (nframes / tps as u64) as usize;
    let limit = pixels
        .and_then(|p| p.max_long_side)
        .unwrap_or(d.video_frame_max_long_side);
    let (sw, sh) = limit_long_side(video.width, video.height, limit);
    let r = d
        .image
        .calc_resize(sw as usize, sh as usize)
        .map_err(|e| e.to_string())?;
    // Timestamps of kept frames at the sampling rate; a fixed frame count
    // spreads the frames over the duration.
    let mut sampling_fps = fps.unwrap_or(d.video_default_fps);
    if fixed.is_some() {
        let duration_s = video.duration_ms as f64 / 1000.0;
        sampling_fps = if duration_s > 0.0 {
            nframes as f64 / duration_s
        } else {
            video.video_fps
        };
    }
    let timestamps = (0..kept)
        .map(|k| (k as f64 * tps as f64) / sampling_fps)
        .collect();
    let total = video.total_frames.unwrap_or_else(|| {
        (video.duration_ms as f64 / 1000.0 * video.video_fps).round_ties_even() as u64
    });
    if total == 0 {
        return Err("the video has no frames".into());
    }
    let frame_indices = (0..nframes)
        .map(|i| {
            ((i as f64 / sampling_fps * video.video_fps).round_ties_even() as u64).min(total - 1)
        })
        .collect();
    Ok(MediaPlan {
        processor: "deepseek_v4_1.v1",
        width: r.best_width as u32,
        height: r.best_height as u32,
        tokens: r.token_count,
        nframes,
        patch_size: 14,
        spatial_merge_size: 3,
        temporal_patch_size: tps,
        frame_indices,
        timestamps,
    })
}

fn plan_video_qwen(
    q: &QwenVision,
    video: &VideoMeta,
    fps: Option<f64>,
    fixed: Option<u32>,
    pixels: Option<&PixelControlOptions>,
) -> Result<MediaPlan, String> {
    let total = match video.total_frames {
        Some(0) => return Err("metadata.total_frames must be > 0".into()),
        Some(t) => t,
        None if video.duration_ms > 0 => {
            (video.duration_ms as f64 / 1000.0 * video.video_fps).round_ties_even() as u64
        }
        None => {
            return Err(
                "metadata needs total_frames, or duration_ms and video_fps to estimate it".into(),
            );
        }
    };
    if total == 0 {
        return Err("the video has no frames".into());
    }
    let n = match fixed {
        Some(n) => {
            if n as u64 > total {
                return Err(format!("nframes {n} exceeds total_frames {total}"));
            }
            n as u64
        }
        None => {
            let target = fps.unwrap_or(q.video_fps);
            let n = (total as f64 / video.video_fps * target) as u64;
            n.max(q.video_min_frames as u64)
                .min(q.video_max_frames as u64)
                .min(total)
        }
    };
    let tps = q.temporal_patch_size.max(1) as u64;
    if n < tps {
        return Err(format!(
            "{n} sampled frames are fewer than the temporal patch size {tps}"
        ));
    }
    // Frame indices: numpy.linspace(0, total - 1, n).round().
    let step = if n > 1 {
        (total - 1) as f64 / (n - 1) as f64
    } else {
        0.0
    };
    let mut indices: Vec<u64> = (0..n)
        .map(|i| {
            if n > 1 && i == n - 1 {
                total - 1
            } else {
                (i as f64 * step).round_ties_even() as u64
            }
        })
        .collect();
    while !(indices.len() as u64).is_multiple_of(tps) {
        indices.push(*indices.last().unwrap_or(&0));
    }
    let seconds: Vec<f64> = indices
        .iter()
        .map(|i| *i as f64 / video.video_fps)
        .collect();
    let timestamps: Vec<f64> = seconds
        .chunks(tps as usize)
        .map(|c| (c[0] + c[c.len() - 1]) / 2.0)
        .collect();
    let (sw, sh) = match pixels.and_then(|p| p.max_long_side) {
        Some(limit) => limit_long_side(video.width, video.height, limit),
        None => (video.width, video.height),
    };
    let factor = q.factor() as u64;
    let mut max_pixels = q.video_max_pixels;
    if let Some(cap) = q.video_max_frame_tokens {
        let frame_cap = cap as u64 * factor * factor;
        let per_frame = frame_cap
            .min(q.video_max_pixels / n)
            .max((q.video_min_pixels as f64 * 1.05) as u64);
        max_pixels = per_frame * n;
    }
    let (h, w) = video_smart_resize(n, sh, sw, tps, q.factor(), q.video_min_pixels, max_pixels)?;
    Ok(MediaPlan {
        processor: "qwen_vl.v1",
        width: w,
        height: h,
        tokens: q.tokens(w, h),
        nframes: indices.len() as u64,
        patch_size: q.patch_size,
        spatial_merge_size: q.merge_size,
        temporal_patch_size: q.temporal_patch_size.max(1),
        frame_indices: indices,
        timestamps,
    })
}

/// Python `round()` of a float division (ties to even).
fn py_round_div(n: f64, d: f64) -> u64 {
    (n / d).round_ties_even().max(0.0) as u64
}

/// Qwen2-VL `smart_resize` of an image: both sides multiples of `factor`,
/// pixels within `[min_pixels, max_pixels]`, aspect ratio kept as closely as
/// possible. Returns (height, width).
pub fn image_smart_resize(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<(u32, u32), String> {
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            h.max(w) / h.min(w)
        ));
    }
    let mut h_bar = py_round_div(h, f) * factor as u64;
    let mut w_bar = py_round_div(w, f) * factor as u64;
    if h_bar * w_bar > max_pixels {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = (factor as u64).max(((h / beta / f).floor() as u64) * factor as u64);
        w_bar = (factor as u64).max(((w / beta / f).floor() as u64) * factor as u64);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = ((h * beta / f).ceil() as u64) * factor as u64;
        w_bar = ((w * beta / f).ceil() as u64) * factor as u64;
    }
    Ok((h_bar as u32, w_bar as u32))
}

/// Qwen3-VL `smart_resize` of a video: the pixel range covers all sampled
/// frames. Returns (height, width) of every frame.
pub fn video_smart_resize(
    num_frames: u64,
    height: u32,
    width: u32,
    temporal_factor: u64,
    factor: u32,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<(u32, u32), String> {
    if num_frames < temporal_factor {
        return Err(format!(
            "{num_frames} frames must be at least the temporal factor {temporal_factor}"
        ));
    }
    let (mut h, mut w, f) = (height as f64, width as f64, factor as f64);
    if h < f || w < f {
        let scale = (f / h).max(f / w);
        h = (h * scale).trunc();
        w = (w * scale).trunc();
    }
    if h.max(w) / h.min(w) > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            h.max(w) / h.min(w)
        ));
    }
    let n = num_frames as f64;
    let mut h_bar = py_round_div(h, f) * factor as u64;
    let mut w_bar = py_round_div(w, f) * factor as u64;
    let t_bar = py_round_div(n, temporal_factor as f64) * temporal_factor;
    if t_bar * h_bar * w_bar > max_pixels {
        let beta = ((n * h * w) / max_pixels as f64).sqrt();
        h_bar = (factor as u64).max(((h / beta / f).floor() as u64) * factor as u64);
        w_bar = (factor as u64).max(((w / beta / f).floor() as u64) * factor as u64);
    } else if t_bar * h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (n * h * w)).sqrt();
        h_bar = ((h * beta / f).ceil() as u64) * factor as u64;
        w_bar = ((w * beta / f).ceil() as u64) * factor as u64;
    }
    Ok((h_bar as u32, w_bar as u32))
}

/// Scale a size down, preserving the aspect ratio, so that its long side is
/// at most `limit` (deepseek-recipe-image `limit_image_mat` rounding).
pub fn limit_long_side(width: u32, height: u32, limit: u32) -> (u32, u32) {
    let long = width.max(height);
    if long <= limit || limit == 0 {
        return (width, height);
    }
    let ratio = limit as f64 / long as f64;
    let w = ((width as f64 * ratio).round() as u32).max(1);
    let h = ((height as f64 * ratio).round() as u32).max(1);
    (w, h)
}

/// Objective properties of a video.
#[derive(Debug, Clone)]
pub struct VideoMeta {
    pub width: u32,
    pub height: u32,
    pub duration_ms: u64,
    pub total_frames: Option<u64>,
    pub video_fps: f64,
}

/// Python `round(n / f) * f` (ties to even).
fn round_by_factor(n: f64, f: u64) -> u64 {
    ((n / f as f64).round_ties_even().max(0.0) as u64) * f
}

/// qwen-vl-utils `smart_nframes`: the number of frames to sample, a multiple
/// of `frame_factor`. A fixed `nframes` wins; otherwise `fps` (default
/// `default_fps`) over the video's frames. Without `total_frames` the count
/// is estimated from `video_fps × duration`.
pub fn smart_nframes(
    video: &VideoMeta,
    fps: Option<f64>,
    nframes: Option<u32>,
    frame_factor: u64,
    default_fps: f64,
) -> Result<u64, String> {
    if video.total_frames == Some(0) {
        return Err("metadata.total_frames must be > 0".into());
    }
    if video.video_fps.is_nan() || video.video_fps <= 0.0 {
        return Err(format!(
            "metadata.video_fps must be > 0, got {}",
            video.video_fps
        ));
    }
    if fps.is_some() && nframes.is_some() {
        return Err("frame_extract_options.fps and nframes are mutually exclusive".into());
    }
    if let Some(f) = fps
        && (f.is_nan() || f <= 0.0)
    {
        return Err(format!("frame_extract_options.fps must be > 0, got {f}"));
    }
    if let Some(n) = nframes {
        let rounded = round_by_factor(n as f64, frame_factor);
        if rounded < frame_factor {
            return Err(format!(
                "nframes {n} (rounded to {rounded}) is below the frame factor {frame_factor}"
            ));
        }
        if let Some(total) = video.total_frames
            && rounded > total
        {
            return Err(format!(
                "nframes {n} (rounded to {rounded}) exceeds total_frames {total}"
            ));
        }
        return Ok(rounded);
    }
    let total = match video.total_frames {
        Some(t) => t,
        None if video.duration_ms > 0 => {
            (video.video_fps * video.duration_ms as f64 / 1000.0).round_ties_even() as u64
        }
        None => {
            return Err(
                "metadata needs total_frames, or duration_ms and video_fps to estimate it".into(),
            );
        }
    };
    let raw = total as f64 / video.video_fps * fps.unwrap_or(default_fps);
    let n = round_by_factor(raw, frame_factor);
    if n < frame_factor || n > total {
        return Err(format!(
            "sampling {raw:.2} frames (rounded to {n}) is outside [{frame_factor}, {total}]"
        ));
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video(total: Option<u64>, fps: f64, ms: u64) -> VideoMeta {
        VideoMeta {
            width: 1280,
            height: 720,
            duration_ms: ms,
            total_frames: total,
            video_fps: fps,
        }
    }

    fn deepseek(spec: &VisionSpec) -> &DeepseekVision {
        match &spec.planner {
            Planner::Deepseek(d) => d,
            Planner::Qwen(_) => unreachable!(),
        }
    }

    #[test]
    fn images_use_the_v41_spec() {
        let s = VisionSpec::deepseek_v41();
        let spec = ImageTokenSpec::v41();
        for (w, h) in [(640, 480), (1920, 1080), (978, 479), (32, 32), (4000, 200)] {
            let p = s.plan_image(w, h, false, None, 1).unwrap();
            let r = spec.calc_resize(w as usize, h as usize).unwrap();
            assert_eq!(p.tokens, r.token_count, "{w}x{h}");
            assert_eq!(
                (p.width, p.height),
                (r.best_width as u32, r.best_height as u32)
            );
        }
    }

    #[test]
    fn long_side_limits_apply_before_the_model_resize() {
        let s = VisionSpec::deepseek_v41();
        let full = s.plan_image(3000, 2000, false, None, 1).unwrap();
        let low = s.plan_image(3000, 2000, true, None, 1).unwrap();
        let (lw, lh) = limit_long_side(3000, 2000, 512);
        assert_eq!((lw, lh), (512, 341));
        assert_eq!(
            low.tokens,
            ImageTokenSpec::v41().calc_token_count(512, 341).unwrap()
        );
        assert!(low.tokens < full.tokens);
        let capped = PixelControlOptions {
            max_long_side: Some(700),
            max_total_pixels: None,
        };
        let p = s.plan_image(3000, 2000, false, Some(&capped), 1).unwrap();
        assert!(p.tokens < full.tokens && p.tokens > low.tokens);
        let tight = PixelControlOptions {
            max_long_side: None,
            max_total_pixels: Some(100_000),
        };
        let err = s
            .plan_image(3000, 2000, false, Some(&tight), 1)
            .unwrap_err();
        assert!(err.contains("max_total_pixels 100000"), "{err}");
    }

    #[test]
    fn source_limits() {
        let s = VisionSpec::deepseek_v41();
        assert!(s.plan_image(0, 10, false, None, 1).is_err());
        assert!(s.plan_image(9000, 10, false, None, 1).is_err());
        assert!(s.plan_image(5000, 4000, false, None, 20).is_err()); // many images
        assert!(
            s.plan_image(4020, 20, false, None, 1)
                .unwrap_err()
                .contains("aspect")
        );
    }

    /// Reference values of qwen-vl-utils' `smart_nframes`, ties included.
    #[test]
    fn smart_nframes_reference_values() {
        let v = video(Some(300), 30.0, 10_000);
        let open = video(None, 24.0, 7_250);
        type Case<'a> = (&'a VideoMeta, Option<f64>, Option<u32>, u64, u64);
        let cases: &[Case] = &[
            (&v, None, None, 1, 20),
            (&v, Some(0.5), None, 2, 4),
            (&v, Some(0.7), None, 1, 7),
            (&v, Some(0.25), None, 1, 2), // 2.5 rounds to even
            (&open, None, None, 1, 14),   // 174 estimated frames, 14.5 -> 14
            (&open, Some(1.0), None, 2, 8),
            (&v, None, Some(7), 2, 8),
            (&v, None, Some(5), 2, 4),
            (&video(None, 30.0, 10_000), None, Some(9), 1, 9),
        ];
        for (i, (meta, fps, n, ff, want)) in cases.iter().enumerate() {
            assert_eq!(
                smart_nframes(meta, *fps, *n, *ff, 2.0),
                Ok(*want),
                "case {i}"
            );
        }
        assert!(smart_nframes(&video(Some(10), 10.0, 1_000), None, Some(12), 2, 2.0).is_err());
        assert!(smart_nframes(&video(Some(3), 30.0, 100), None, None, 2, 2.0).is_err());
        assert!(smart_nframes(&v, Some(1.0), Some(4), 1, 2.0).is_err());
        assert!(smart_nframes(&video(Some(300), 0.0, 10_000), None, None, 1, 2.0).is_err());
    }

    #[test]
    fn video_plans_sample_and_cap_frames() {
        let s = VisionSpec::deepseek_v41();
        let v = video(Some(300), 30.0, 10_000);
        let p = s.plan_video(&v, None, 1).unwrap();
        assert_eq!((p.frame_indices.len(), p.groups()), (20, 20));
        assert_eq!(p.nframes, 20);
        assert_eq!(&p.frame_indices[..4], &[0, 15, 30, 45]);
        assert_eq!(p.timestamps.len(), 20);
        assert!((p.timestamps[3] - 1.5).abs() < 1e-9);
        // Frames default to a 672-pixel long side before the model resize.
        let (w, h) = limit_long_side(1280, 720, 672);
        assert_eq!(
            p.tokens,
            deepseek(&s)
                .image
                .calc_token_count(w as usize, h as usize)
                .unwrap()
        );
        // A fixed frame count spreads the timestamps over the duration.
        let fixed = FrameExtractOptions {
            nframes: Some(4),
            ..Default::default()
        };
        let p = s.plan_video(&v, Some(&fixed), 1).unwrap();
        assert_eq!(p.timestamps, vec![0.0, 2.5, 5.0, 7.5]);
        let tight = FrameExtractOptions {
            fps: Some(1.0),
            nframes: None,
            pixel_control_options: Some(PixelControlOptions {
                max_long_side: Some(336),
                max_total_pixels: Some(1_000_000),
            }),
        };
        let err = s.plan_video(&v, Some(&tight), 1).unwrap_err();
        assert!(err.contains("exceeds max_total_pixels 1000000"), "{err}");
    }

    /// Reference values of transformers' Qwen2-VL / Qwen3-VL `smart_resize`.
    #[test]
    fn qwen_smart_resize_reference_values() {
        assert_eq!(
            image_smart_resize(1080, 1920, 32, 65_536, 16_777_216),
            Ok((1088, 1920))
        );
        assert_eq!(
            image_smart_resize(200, 300, 32, 65_536, 16_777_216),
            Ok((224, 320))
        );
        // Below the minimum: scaled up.
        assert_eq!(
            image_smart_resize(30, 20, 32, 65_536, 16_777_216),
            Ok((320, 224))
        );
        // Above the maximum (Qwen2.5-VL settings): scaled down.
        assert_eq!(
            image_smart_resize(8000, 6000, 28, 3_136, 12_845_056),
            Ok((4116, 3080))
        );
        assert!(image_smart_resize(10, 2500, 32, 65_536, 16_777_216).is_err());
        assert_eq!(
            video_smart_resize(10, 720, 1280, 2, 32, 4_096, 7_864_320),
            Ok((640, 1152))
        );
        assert!(video_smart_resize(1, 720, 1280, 2, 32, 4_096, 7_864_320).is_err());
    }

    #[test]
    fn qwen_video_plans_follow_qwen3_vl() {
        let s = VisionSpec::qwen(QwenVision::qwen3_vl());
        let v = video(Some(300), 30.0, 10_000);
        let one_fps = FrameExtractOptions {
            fps: Some(1.0),
            ..Default::default()
        };
        let p = s.plan_video(&v, Some(&one_fps), 1).unwrap();
        assert_eq!((p.frame_indices.len(), p.groups()), (10, 5));
        assert_eq!(p.nframes, 10);
        assert_eq!((p.width, p.height, p.tokens), (1152, 640, 720));
        assert_eq!(p.timestamps.len(), 5);
        assert!((p.timestamps[0] - 0.55).abs() < 1e-12);
        // An odd frame count repeats the last frame in the last temporal patch.
        let five = FrameExtractOptions {
            nframes: Some(5),
            ..Default::default()
        };
        let short = VideoMeta {
            width: 640,
            height: 360,
            duration_ms: 2500,
            total_frames: None,
            video_fps: 24.0,
        };
        let p = s.plan_video(&short, Some(&five), 1).unwrap();
        assert_eq!((p.frame_indices.len(), p.groups()), (6, 3));
        assert_eq!(p.nframes, 6); // includes the explicitly repeated last frame
        assert_eq!(&p.frame_indices[4..], &[59, 59]);
        assert_eq!((p.width, p.height, p.tokens), (640, 352, 220));
        // Too few frames for one temporal patch.
        let one = VideoMeta {
            total_frames: Some(1),
            ..short.clone()
        };
        assert!(s.plan_video(&one, None, 1).is_err());
    }

    #[test]
    fn processor_configs_override_defaults() {
        let mut q = QwenVision::qwen3_vl();
        q.apply_processor_config(
            &serde_json::json!({"patch_size": 14, "min_pixels": 3136, "max_pixels": 1003520}),
            false,
        );
        assert_eq!(
            (q.patch_size, q.image_min_pixels, q.image_max_pixels),
            (14, 3136, 1003520)
        );
        q.apply_processor_config(
            &serde_json::json!({"size": {"shortest_edge": 4096, "longest_edge": 1000000}, "fps": 1.0, "max_frames": 64}),
            true,
        );
        assert_eq!(
            (
                q.video_min_pixels,
                q.video_max_pixels,
                q.video_fps,
                q.video_max_frames
            ),
            (4096, 1000000, 1.0, 64)
        );
    }
}
