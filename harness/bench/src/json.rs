//! Strict RFC 8259 JSON reader.
//!
//! The schema for benchmark receipts is deliberately closed, so the reader is
//! deliberately strict: no comments, no trailing commas, no duplicate object
//! keys, no leading `+`, no leading zeroes, no `NaN`/`Infinity`, no raw control
//! characters in strings, and no data after the top-level value. Every
//! rejection carries the byte offset of the problem.
//!
//! This is a small reader for this crate's own schema, not a general-purpose
//! JSON library.

use std::fmt;

/// Deepest accepted container nesting. Receipts are shallow; the limit keeps a
/// hostile document from exhausting the stack.
const MAX_DEPTH: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub offset: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "malformed JSON at byte {}: {}",
            self.offset, self.message
        )
    }
}

/// Parse exactly one JSON value. Trailing non-whitespace is an error.
pub fn parse(text: &str) -> Result<Json, ParseError> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        pos: 0,
    };
    parser.skip_ws();
    let value = parser.value(0)?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(parser.error("trailing data after the top-level value"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn error(&self, message: impl Into<String>) -> ParseError {
        ParseError {
            offset: self.pos,
            message: message.into(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.pos += 1;
        Some(byte)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, byte: u8, what: &str) -> Result<(), ParseError> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.error(format!("expected {what}")))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, ParseError> {
        if depth > MAX_DEPTH {
            return Err(self.error(format!("nesting deeper than {MAX_DEPTH} levels")));
        }
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => {
                self.literal("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(Json::Bool(false))
            }
            Some(b'n') => {
                self.literal("null")?;
                Ok(Json::Null)
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            None => Err(self.error("unexpected end of input")),
            Some(_) => Err(self.error("unexpected character where a value was expected")),
        }
    }

    fn literal(&mut self, literal: &str) -> Result<(), ParseError> {
        if self.bytes[self.pos..].starts_with(literal.as_bytes()) {
            self.pos += literal.len();
            Ok(())
        } else {
            Err(self.error(format!("invalid literal, expected {literal:?}")))
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, ParseError> {
        self.expect(b'{', "'{'")?;
        let mut entries: Vec<(String, Json)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(entries));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.error("expected a quoted object key"));
            }
            let key_offset = self.pos;
            let key = self.string()?;
            if entries.iter().any(|(existing, _)| existing == &key) {
                return Err(ParseError {
                    offset: key_offset,
                    message: format!("duplicate object key {key:?}"),
                });
            }
            self.skip_ws();
            self.expect(b':', "':' after an object key")?;
            self.skip_ws();
            let value = self.value(depth + 1)?;
            entries.push((key, value));
            self.skip_ws();
            match self.bump() {
                Some(b',') => {
                    self.skip_ws();
                    if self.peek() == Some(b'}') {
                        return Err(self.error("trailing comma in object"));
                    }
                }
                Some(b'}') => break,
                Some(_) => return Err(self.error("expected ',' or '}' in object")),
                None => return Err(self.error("unterminated object")),
            }
        }
        Ok(Json::Obj(entries))
    }

    fn array(&mut self, depth: usize) -> Result<Json, ParseError> {
        self.expect(b'[', "'['")?;
        let mut items: Vec<Json> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.bump() {
                Some(b',') => {
                    self.skip_ws();
                    if self.peek() == Some(b']') {
                        return Err(self.error("trailing comma in array"));
                    }
                }
                Some(b']') => break,
                Some(_) => return Err(self.error("expected ',' or ']' in array")),
                None => return Err(self.error("unterminated array")),
            }
        }
        Ok(Json::Arr(items))
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.expect(b'"', "a string")?;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(byte) = self.bump() else {
                return Err(self.error("unterminated string"));
            };
            match byte {
                b'"' => break,
                b'\\' => {
                    let Some(escaped) = self.bump() else {
                        return Err(self.error("unterminated escape sequence"));
                    };
                    match escaped {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0c),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let first = self.hex4()?;
                            let character = if (0xD800..=0xDBFF).contains(&first) {
                                self.expect(b'\\', "'\\\\' starting the low surrogate")?;
                                self.expect(b'u', "'u' starting the low surrogate")?;
                                let second = self.hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&second) {
                                    return Err(self
                                        .error("high surrogate not followed by a low surrogate"));
                                }
                                let scalar = 0x1_0000
                                    + (((u32::from(first) - 0xD800) << 10)
                                        | (u32::from(second) - 0xDC00));
                                char::from_u32(scalar)
                                    .ok_or_else(|| self.error("invalid unicode scalar"))?
                            } else if (0xDC00..=0xDFFF).contains(&first) {
                                return Err(self.error("lone low surrogate in string escape"));
                            } else {
                                char::from_u32(u32::from(first))
                                    .ok_or_else(|| self.error("invalid unicode scalar"))?
                            };
                            let mut buffer = [0u8; 4];
                            out.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
                        }
                        _ => return Err(self.error("invalid escape sequence")),
                    }
                }
                0x00..=0x1f => {
                    return Err(self.error("unescaped control character in string"));
                }
                _ => out.push(byte),
            }
        }
        String::from_utf8(out).map_err(|_| self.error("string is not valid UTF-8"))
    }

    fn hex4(&mut self) -> Result<u16, ParseError> {
        let mut value = 0u16;
        for _ in 0..4 {
            let Some(byte) = self.bump() else {
                return Err(self.error("truncated \\u escape"));
            };
            let digit = match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => return Err(self.error("invalid hex digit in \\u escape")),
            };
            value = value * 16 + u16::from(digit);
        }
        Ok(value)
    }

    fn number(&mut self) -> Result<Json, ParseError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.error("leading zero in number"));
                }
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(self.error("invalid number: expected a digit")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("invalid number: expected a digit after '.'"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("invalid number: expected an exponent digit"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let raw = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.error("number is not UTF-8"))?;
        let value: f64 = raw
            .parse()
            .map_err(|_| self.error("invalid number representation"))?;
        if !value.is_finite() {
            return Err(self.error("number is outside the finite range"));
        }
        Ok(Json::Num(value))
    }
}

/// Serialise a value. Numbers use Rust's shortest round-trip formatting, so
/// `parse(&write(v)) == v` for every finite value this crate produces.
pub fn write(value: &Json) -> String {
    let mut out = String::new();
    write_into(&mut out, value);
    out
}

fn write_into(out: &mut String, value: &Json) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => {
            if n.fract() == 0.0 && n.abs() < 9_007_199_254_740_992.0 {
                out.push_str(&format!("{}", *n as i64));
            } else {
                out.push_str(&format!("{n}"));
            }
        }
        Json::Str(s) => write_str(out, s),
        Json::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_into(out, item);
            }
            out.push(']');
        }
        Json::Obj(entries) => {
            out.push('{');
            for (i, (key, item)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_str(out, key);
                out.push(':');
                write_into(out, item);
            }
            out.push('}');
        }
    }
}

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::{parse, Json};

    #[test]
    fn parses_nested_document_and_escapes() {
        let value = parse(
            "{\"a\": [1, -2.5, 3e2], \"b\": {\"c\": \"x\\n\\u00e9\\ud83d\\ude00\", \"d\": true, \"e\": null}}",
        )
        .expect("valid document");
        let Json::Obj(entries) = value else {
            panic!("expected object");
        };
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn rejects_trailing_comma_and_leading_zero() {
        assert!(parse("{\"a\": 1,}").is_err());
        assert!(parse("[1, 2,]").is_err());
        assert!(parse("{\"a\": 01}").is_err());
        assert!(parse("{\"a\": 1} x").is_err());
    }

    #[test]
    fn rejects_duplicate_keys() {
        let error = parse("{\"a\": 1, \"a\": 2}").expect_err("duplicate key must fail");
        assert!(error.message.contains("duplicate object key"), "{error}");
    }

    #[test]
    fn rejects_non_json_numbers_and_control_characters() {
        assert!(parse("{\"a\": NaN}").is_err());
        assert!(parse("{\"a\": 1e999}").is_err());
        assert!(parse("{\"a\": \"raw\nnewline\"}").is_err());
        assert!(parse("{\"a\": 1.}").is_err());
        assert!(parse("{\"a\": +1}").is_err());
    }

    #[test]
    fn rejects_lone_surrogate_and_bad_escape() {
        assert!(parse("{\"a\": \"\\ud800\"}").is_err());
        assert!(parse("{\"a\": \"\\q\"}").is_err());
        assert!(parse("{\"a\": \"\\u12\"}").is_err());
    }

    #[test]
    fn rejects_excessive_nesting() {
        let document = format!("{}{}", "[".repeat(80), "]".repeat(80));
        assert!(parse(&document).is_err());
    }
}
