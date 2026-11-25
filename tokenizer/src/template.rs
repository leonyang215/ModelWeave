//! Hugging Face chat templates rendered with minijinja the way transformers'
//! `apply_chat_template` renders them with Jinja2: `trim_blocks` and
//! `lstrip_blocks`, Python string methods (`split`, `strip`, `startswith`,
//! ...), the transformers `tojson` filter (Python `json.dumps` spacing,
//! `ensure_ascii=False`, `indent`, `separators`, `sort_keys`),
//! `raise_exception`, `strftime_now`, and Python spellings of printed
//! values (`None`, `True`, `1e-05`).

use std::fmt::Write as _;
use std::io;

use minijinja::syntax::SyntaxConfig;
use minijinja::value::{Kwargs, Serde, Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind, Output, State};
use serde::Serialize;
use serde_json::ser::Formatter;

/// How a template reads message content (vLLM's chat template content
/// format): a plain string, or OpenAI-style lists of typed parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentFormat {
    String,
    Parts,
}

impl ContentFormat {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "string" => Ok(Self::String),
            "openai" | "parts" => Ok(Self::Parts),
            other => Err(format!(
                "unknown content format {other:?} (want string or openai)"
            )),
        }
    }

    /// Detect the format from the template source: templates that loop over
    /// a message's content take parts.
    pub fn detect(source: &str) -> Self {
        let mut rest = source;
        while let Some(i) = rest.find("for ") {
            rest = &rest[i + 4..];
            let head = &rest[..rest.find("%}").unwrap_or(rest.len()).min(rest.len())];
            if let Some(j) = head.find(" in ") {
                let iterable = head[j + 4..].trim();
                if iterable.contains("content") {
                    return Self::Parts;
                }
            }
        }
        Self::String
    }
}

/// Error of a render.
#[derive(Debug)]
pub struct TemplateError(pub String);

/// A compiled chat template.
pub struct ChatTemplate {
    env: Environment<'static>,
    pub content_format: ContentFormat,
}

const NAME: &str = "chat_template";

impl ChatTemplate {
    pub fn new(source: String) -> Result<Self, String> {
        let content_format = ContentFormat::detect(&source);
        let mut env = Environment::new();
        env.set_syntax(
            SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|e| e.to_string())?,
        );
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.set_formatter(py_formatter);
        env.add_filter("tojson", tojson);
        env.add_filter("string", |v: Value| py_str(&v));
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned(NAME, source)
            .map_err(|e| format!("invalid chat template: {e:#}"))?;
        Ok(Self {
            env,
            content_format,
        })
    }

    /// Render with the given variables.
    pub fn render(&self, vars: &serde_json::Value) -> Result<String, TemplateError> {
        let tmpl = self
            .env
            .get_template(NAME)
            .map_err(|e| TemplateError(e.to_string()))?;
        tmpl.render(Value::from(Serde(vars)))
            .map_err(|e| TemplateError(template_error_message(&e)))
    }
}

fn template_error_message(e: &Error) -> String {
    match e.kind() {
        // raise_exception: the template's own validation message.
        ErrorKind::InvalidOperation => e.detail().unwrap_or("invalid operation").to_string(),
        _ => format!("{e}"),
    }
}

fn raise_exception(message: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

// ---------------------------------------------------------------------------
// Python-style printing
// ---------------------------------------------------------------------------

fn py_formatter(out: &mut Output, _state: &mut State, value: &Value) -> Result<(), Error> {
    if value.is_undefined() {
        return Ok(());
    }
    out.write_str(&py_str(value))
        .map_err(|_| Error::new(ErrorKind::WriteFailure, "write failed"))
}

/// Python `str()` of a template value.
pub fn py_str(v: &Value) -> String {
    match v.kind() {
        ValueKind::String => v.as_str().unwrap_or_default().to_string(),
        ValueKind::Undefined => String::new(),
        _ => py_repr(v),
    }
}

/// Python `repr()` of a template value.
fn py_repr(v: &Value) -> String {
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => "None".into(),
        ValueKind::Bool => if v.is_true() { "True" } else { "False" }.into(),
        ValueKind::Number => {
            if v.is_integer() {
                v.to_string()
            } else {
                match f64::try_from(v.clone()) {
                    Ok(f) => py_float(f),
                    Err(_) => v.to_string(),
                }
            }
        }
        ValueKind::String => py_quote(v.as_str().unwrap_or_default()),
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<String> = v
                .try_iter()
                .map(|it| it.map(|x| py_repr(&x)).collect())
                .unwrap_or_default();
            format!("[{}]", items.join(", "))
        }
        ValueKind::Map => {
            let mut parts = Vec::new();
            if let Ok(keys) = v.try_iter() {
                for k in keys {
                    let val = v.get_item(&k).unwrap_or_default();
                    parts.push(format!("{}: {}", py_repr(&k), py_repr(&val)));
                }
            }
            format!("{{{}}}", parts.join(", "))
        }
        _ => v.to_string(),
    }
}

fn py_quote(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `repr()` of a float: the shortest round-trip digits, exponent form
/// below 1e-4 and from 1e16 on, written `1e-05` / `1e+16`.
pub fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let s = format!("{f:?}");
    match s.find('e') {
        None => s,
        Some(i) => {
            let (mant, exp) = (&s[..i], &s[i + 1..]);
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', exp),
            };
            let mant = mant.strip_suffix(".0").unwrap_or(mant);
            format!("{mant}e{sign}{digits:0>2}")
        }
    }
}

// ---------------------------------------------------------------------------
// tojson: Python json.dumps
// ---------------------------------------------------------------------------

/// Serializer formatting of Python's `json.dumps`.
struct PyJson {
    indent: Option<Vec<u8>>,
    item_sep: Vec<u8>,
    key_sep: Vec<u8>,
    ensure_ascii: bool,
    level: usize,
    has_value: bool,
}

impl PyJson {
    fn new(
        indent: Option<Vec<u8>>,
        separators: Option<(String, String)>,
        ensure_ascii: bool,
    ) -> Self {
        let (item, key) = separators.unwrap_or_else(|| {
            let item = if indent.is_some() { "," } else { ", " };
            (item.to_string(), ": ".to_string())
        });
        Self {
            indent,
            item_sep: item.into_bytes(),
            key_sep: key.into_bytes(),
            ensure_ascii,
            level: 0,
            has_value: false,
        }
    }

    fn newline<W: ?Sized + io::Write>(&self, w: &mut W) -> io::Result<()> {
        if let Some(ind) = &self.indent {
            w.write_all(b"\n")?;
            for _ in 0..self.level {
                w.write_all(ind)?;
            }
        }
        Ok(())
    }

    fn begin_item<W: ?Sized + io::Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
        if !first {
            w.write_all(&self.item_sep)?;
        }
        self.newline(w)
    }

    fn open<W: ?Sized + io::Write>(&mut self, w: &mut W, b: &[u8]) -> io::Result<()> {
        self.level += 1;
        self.has_value = false;
        w.write_all(b)
    }

    fn close<W: ?Sized + io::Write>(&mut self, w: &mut W, b: &[u8]) -> io::Result<()> {
        self.level -= 1;
        if self.has_value {
            self.newline(w)?;
        }
        self.has_value = true;
        w.write_all(b)
    }
}

impl Formatter for PyJson {
    fn write_f64<W: ?Sized + io::Write>(&mut self, w: &mut W, value: f64) -> io::Result<()> {
        w.write_all(py_float(value).as_bytes())
    }

    fn write_f32<W: ?Sized + io::Write>(&mut self, w: &mut W, value: f32) -> io::Result<()> {
        self.write_f64(w, value as f64)
    }

    fn write_string_fragment<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        if !self.ensure_ascii {
            return w.write_all(fragment.as_bytes());
        }
        let mut buf = [0u16; 2];
        for c in fragment.chars() {
            if c.is_ascii() {
                w.write_all(&[c as u8])?;
            } else {
                for unit in c.encode_utf16(&mut buf) {
                    write!(w, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }

    fn begin_array<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.open(w, b"[")
    }

    fn end_array<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.close(w, b"]")
    }

    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.begin_item(w, first)
    }

    fn end_array_value<W: ?Sized + io::Write>(&mut self, _w: &mut W) -> io::Result<()> {
        self.has_value = true;
        Ok(())
    }

    fn begin_object<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.open(w, b"{")
    }

    fn end_object<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.close(w, b"}")
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.begin_item(w, first)
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(&self.key_sep)
    }

    fn end_object_value<W: ?Sized + io::Write>(&mut self, _w: &mut W) -> io::Result<()> {
        self.has_value = true;
        Ok(())
    }
}

fn dump<T: Serialize + ?Sized>(value: &T, f: PyJson) -> Result<String, String> {
    let mut buf = Vec::with_capacity(256);
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, f);
    value.serialize(&mut ser).map_err(|e| e.to_string())?;
    String::from_utf8(buf).map_err(|e| e.to_string())
}

fn sort_keys(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(m) => {
            let mut entries: Vec<_> = m.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            serde_json::Value::Object(
                entries
                    .into_iter()
                    .map(|(k, v)| (k, sort_keys(v)))
                    .collect(),
            )
        }
        serde_json::Value::Array(a) => {
            serde_json::Value::Array(a.into_iter().map(sort_keys).collect())
        }
        other => other,
    }
}

fn invalid(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidOperation, msg.into())
}

/// transformers' `tojson(x, ensure_ascii=False, indent=None, separators=None,
/// sort_keys=False)`.
fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<Value> = kwargs.get("indent")?;
    let separators: Option<Value> = kwargs.get("separators")?;
    let sort: Option<bool> = kwargs.get("sort_keys")?;
    kwargs.assert_all_used()?;
    let indent = match indent {
        None => None,
        Some(v) => match v.kind() {
            ValueKind::None | ValueKind::Undefined => None,
            ValueKind::Number if v.is_integer() => {
                let n = v.as_i64().unwrap_or(0).max(0) as usize;
                Some(vec![b' '; n])
            }
            ValueKind::String => Some(v.as_str().unwrap_or_default().as_bytes().to_vec()),
            other => {
                return Err(invalid(format!(
                    "tojson: indent must be an integer or a string, not {other}"
                )));
            }
        },
    };
    let separators = match separators {
        None => None,
        Some(v) if v.is_none() || v.is_undefined() => None,
        Some(v) => {
            let items: Vec<Value> = v.try_iter().map(|i| i.collect()).unwrap_or_default();
            match items.as_slice() {
                [a, b] if a.as_str().is_some() && b.as_str().is_some() => Some((
                    a.as_str().unwrap_or_default().to_string(),
                    b.as_str().unwrap_or_default().to_string(),
                )),
                _ => return Err(invalid("tojson: separators must be a pair of strings")),
            }
        }
    };
    let f = PyJson::new(indent, separators, ensure_ascii.unwrap_or(false));
    let out = if sort.unwrap_or(false) {
        let json = serde_json::to_value(&value).map_err(|e| invalid(e.to_string()))?;
        dump(&sort_keys(json), f)
    } else {
        dump(&value, f)
    }
    .map_err(invalid)?;
    Ok(Value::from_safe_string(out))
}

// ---------------------------------------------------------------------------
// strftime_now
// ---------------------------------------------------------------------------

/// The current UTC time formatted with a subset of C `strftime`
/// (`%Y %y %m %d %e %H %I %M %S %p %b %B %a %A %j %%`).
fn strftime_now(format: String) -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    strftime(secs, &format)
}

fn strftime(unix: i64, format: &str) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const DAYS: [&str; 7] = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ];
    let days = unix.div_euclid(86_400);
    let sod = unix.rem_euclid(86_400);
    let (hour, minute, second) = (sod / 3600, sod % 3600 / 60, sod % 60);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy_mar = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy_mar + 2) / 153;
    let day = doy_mar - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let weekday = (days + 3).rem_euclid(7) as usize; // 1970-01-01 was a Thursday
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let yday = cum[(month - 1) as usize] + day + i64::from(leap && month > 2);
    let mut out = String::new();
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => {
                let _ = write!(out, "{year}");
            }
            Some('y') => {
                let _ = write!(out, "{:02}", year % 100);
            }
            Some('m') => {
                let _ = write!(out, "{month:02}");
            }
            Some('d') => {
                let _ = write!(out, "{day:02}");
            }
            Some('e') => {
                let _ = write!(out, "{day:>2}");
            }
            Some('H') => {
                let _ = write!(out, "{hour:02}");
            }
            Some('I') => {
                let _ = write!(out, "{:02}", if hour % 12 == 0 { 12 } else { hour % 12 });
            }
            Some('M') => {
                let _ = write!(out, "{minute:02}");
            }
            Some('S') => {
                let _ = write!(out, "{second:02}");
            }
            Some('p') => out.push_str(if hour < 12 { "AM" } else { "PM" }),
            Some('B') => out.push_str(MONTHS[(month - 1) as usize]),
            Some('b') => out.push_str(&MONTHS[(month - 1) as usize][..3]),
            Some('A') => out.push_str(DAYS[weekday]),
            Some('a') => out.push_str(&DAYS[weekday][..3]),
            Some('j') => {
                let _ = write!(out, "{yday:03}");
            }
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn render(src: &str, vars: serde_json::Value) -> String {
        ChatTemplate::new(src.to_string())
            .unwrap()
            .render(&vars)
            .unwrap()
    }

    #[test]
    fn tojson_matches_python_json_dumps() {
        let v = json!({"name": "get_weather", "args": [1, 2.5, {"a": true, "b": null}], "city": "广州", "q": "a \"b\" \\ </x>\n"});
        assert_eq!(
            render("{{ v | tojson }}", json!({"v": v})),
            r#"{"name": "get_weather", "args": [1, 2.5, {"a": true, "b": null}], "city": "广州", "q": "a \"b\" \\ </x>\n"}"#
        );
        assert_eq!(
            render("{{ v | tojson(ensure_ascii=True) }}", json!({"v": "广🙂"})),
            r#""\u5e7f\ud83d\ude42""#
        );
        assert_eq!(
            render(
                "{{ v | tojson(indent=2) }}",
                json!({"v": {"a": [1, {}], "b": []}})
            ),
            "{\n  \"a\": [\n    1,\n    {}\n  ],\n  \"b\": []\n}"
        );
        assert_eq!(
            render(
                "{{ v | tojson(separators=(',', ':'), sort_keys=true) }}",
                json!({"v": {"b": 1, "a": 2}})
            ),
            r#"{"a":2,"b":1}"#
        );
        for (f, want) in [
            (1e20, "1e+20"),
            (1e-5, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (100.0, "100.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
        ] {
            assert_eq!(py_float(f), want, "{f}");
        }
    }

    #[test]
    fn python_semantics() {
        // trim_blocks / lstrip_blocks, Python string methods, Python printing
        // (expected outputs of Jinja2 with transformers' settings).
        let src = "{%- for m in messages[::-1] %}\n  {{ m.content.split('|')[0].strip() }}{% if not loop.last %},{% endif %}\n{%- endfor %}";
        assert_eq!(
            render(
                src,
                json!({"messages": [{"content": " a |x"}, {"content": "b|y"}]})
            ),
            "  b,  a"
        );
        assert_eq!(
            render(
                "{{ x }} {{ y }} {{ z }} {{ w }} {{ none_var }}|",
                json!({"x": null, "y": true, "z": 1.0, "w": ["a", "b'c"]})
            ),
            "None True 1.0 ['a', \"b'c\"] |"
        );
        assert_eq!(
            render(
                "{{ 'X'.startswith('X') }}{{ 'ab'.endswith('c') }}",
                json!({})
            ),
            "TrueFalse"
        );
        let ns = "{%- set ns = namespace(n=0) %}{%- for i in [1, 2, 3] %}{%- set ns.n = ns.n + i %}{%- endfor %}{{ ns.n }}";
        assert_eq!(render(ns, json!({})), "6");
    }

    #[test]
    fn raise_exception_surfaces_the_message() {
        let t =
            ChatTemplate::new("{{ raise_exception('Conversation roles must alternate') }}".into())
                .unwrap();
        let err = t.render(&json!({})).unwrap_err();
        assert!(
            err.0.contains("Conversation roles must alternate"),
            "{}",
            err.0
        );
    }

    #[test]
    fn content_format_detection() {
        assert_eq!(
            ContentFormat::detect("{% for m in messages %}{{ m.content }}{% endfor %}"),
            ContentFormat::String
        );
        assert_eq!(
            ContentFormat::detect(
                "{% for m in messages %}{% for content in m.content %}{{ content.text }}{% endfor %}{% endfor %}"
            ),
            ContentFormat::Parts
        );
        assert_eq!(
            ContentFormat::detect("{%- for item in message['content'] -%}x{%- endfor -%}"),
            ContentFormat::Parts
        );
    }

    #[test]
    fn strftime_formats_utc_dates() {
        // 2026-11-26T09:05:07Z is a Thursday, day 330 of the year.
        let t = 1_795_683_907;
        assert_eq!(
            strftime(t, "%d %b %Y %H:%M:%S %a %j %%"),
            "26 Nov 2026 09:05:07 Thu 330 %"
        );
        assert_eq!(strftime(0, "%Y-%m-%d %A"), "1970-01-01 Thursday");
        assert_eq!(strftime(951_782_400, "%B %e"), "February 29"); // leap day 2000
    }
}
