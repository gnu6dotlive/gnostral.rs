//! Cut streamed text into pieces a text-to-speech model can say on their own.
//!
//! A piece ends after `.`, `!`, `?`, `;`, `:` or a newline when the next
//! character is whitespace (so `3.14`, `e.g.x` and `www.site` stay whole), or at
//! the last space once `max_chars` is reached so a long sentence does not delay
//! speech. Pieces are emitted as soon as they are known; [`Chunker::finish`]
//! flushes the rest. Concatenating every piece gives back the input exactly.

#[derive(Debug, Clone)]
pub struct Chunker {
    max_chars: usize,
    pending: String,
}

const ENDERS: [char; 6] = ['.', '!', '?', ';', ':', '\n'];

impl Chunker {
    pub fn new(max_chars: usize) -> Self {
        assert!(max_chars > 0, "max_chars must be positive");
        Self {
            max_chars,
            pending: String::new(),
        }
    }

    /// Feed streamed text; returns the pieces completed by it.
    pub fn push(&mut self, text: &str) -> Vec<String> {
        self.pending.push_str(text);
        let mut out = Vec::new();
        while let Some(end) = self.cut() {
            let rest = self.pending.split_off(end);
            out.push(std::mem::replace(&mut self.pending, rest));
        }
        out
    }

    /// The text not yet emitted, as a final piece (if any).
    pub fn finish(&mut self) -> Option<String> {
        (!self.pending.is_empty()).then(|| std::mem::take(&mut self.pending))
    }

    /// Byte index where the next piece ends, including its trailing whitespace.
    fn cut(&self) -> Option<usize> {
        let chars: Vec<(usize, char)> = self.pending.char_indices().collect();
        for (n, &(_, c)) in chars.iter().enumerate() {
            if n >= self.max_chars {
                break;
            }
            if ENDERS.contains(&c) {
                let next = chars.get(n + 1).map(|&(_, c)| c);
                if c == '\n' || next.is_some_and(char::is_whitespace) {
                    return Some(self.end_after_spaces(&chars, n + 1));
                }
            }
        }
        if chars.len() <= self.max_chars {
            return None;
        }
        // Too long without a sentence end: cut at the last space inside the limit, or hard.
        let space = chars[..self.max_chars].iter().rposition(|&(_, c)| c == ' ');
        Some(match space {
            Some(n) if n > 0 => self.end_after_spaces(&chars, n),
            _ => chars[self.max_chars].0,
        })
    }

    fn end_after_spaces(&self, chars: &[(usize, char)], mut n: usize) -> usize {
        while n < chars.len() && chars[n].1.is_whitespace() {
            n += 1;
        }
        chars.get(n).map_or(self.pending.len(), |&(i, _)| i)
    }
}
