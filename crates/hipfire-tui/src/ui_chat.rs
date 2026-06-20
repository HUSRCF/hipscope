// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Chat message body rendering — the one fiddly bit, isolated and pure.
//!
//! Turns a message's text into styled ratatui lines, giving fenced ``` code
//! blocks a distinct background, dimming trailing comments, and tinting inline
//! `code` spans in prose. Deliberately light (no per-language grammar): a code
//! TUI wants legible structure, not a full highlighter. Pure + color-injected so
//! it is unit-testable without a terminal.

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

/// Colors injected by the caller (from the UI theme) so this module stays pure.
#[derive(Clone, Copy)]
pub struct CodeTheme {
    pub text: Color,
    pub code_fg: Color,
    pub code_bg: Color,
    pub comment: Color,
    pub fence: Color,
}

/// Render a message body into styled lines. Fenced code blocks (delimited by a
/// line whose first non-space chars are ```` ``` ````) get the code background;
/// inside them, a trailing `//` or `#` comment is dimmed. Outside code, inline
/// `` `code` `` spans are tinted.
pub fn render_body(content: &str, theme: &CodeTheme) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut in_code = false;
    for raw in content.lines() {
        if raw.trim_start().starts_with("```") {
            in_code = !in_code;
            out.push(fence_line(raw, in_code, theme));
        } else if in_code {
            out.push(code_line(raw, theme));
        } else {
            out.push(prose_line(raw, theme));
        }
    }
    out
}

fn fence_line(raw: &str, opening: bool, theme: &CodeTheme) -> Line<'static> {
    let lang = raw.trim_start().trim_start_matches('`').trim();
    let label = if opening && !lang.is_empty() {
        format!("``` {lang}")
    } else {
        "```".to_string()
    };
    Line::from(Span::styled(
        label,
        Style::default().fg(theme.fence).add_modifier(Modifier::DIM),
    ))
}

fn code_line(raw: &str, theme: &CodeTheme) -> Line<'static> {
    let code = Style::default().fg(theme.code_fg).bg(theme.code_bg);
    match find_comment(raw) {
        Some(idx) => Line::from(vec![
            Span::styled(raw[..idx].to_string(), code),
            Span::styled(
                raw[idx..].to_string(),
                Style::default().fg(theme.comment).bg(theme.code_bg),
            ),
        ]),
        None => Line::from(Span::styled(raw.to_string(), code)),
    }
}

fn prose_line(raw: &str, theme: &CodeTheme) -> Line<'static> {
    let text = Style::default().fg(theme.text);
    if !raw.contains('`') {
        return Line::from(Span::styled(raw.to_string(), text));
    }
    let code = Style::default().fg(theme.code_fg).bg(theme.code_bg);
    let mut spans = Vec::new();
    let mut is_code = false;
    for part in raw.split('`') {
        if !part.is_empty() {
            spans.push(Span::styled(
                part.to_string(),
                if is_code { code } else { text },
            ));
        }
        is_code = !is_code;
    }
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), text));
    }
    Line::from(spans)
}

/// Byte index of a line-trailing `//` or `#` comment, skipping matches inside
/// single/double-quoted strings. None if the line has no comment.
fn find_comment(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut in_str: Option<u8> = None;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match in_str {
            Some(q) => {
                if c == q && (i == 0 || bytes[i - 1] != b'\\') {
                    in_str = None;
                }
            }
            None => {
                if c == b'"' || c == b'\'' {
                    in_str = Some(c);
                } else if c == b'#' {
                    return Some(i);
                } else if c == b'/' && bytes.get(i + 1) == Some(&b'/') {
                    return Some(i);
                }
            }
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> CodeTheme {
        CodeTheme {
            text: Color::White,
            code_fg: Color::Cyan,
            code_bg: Color::Black,
            comment: Color::DarkGray,
            fence: Color::Blue,
        }
    }

    #[test]
    fn code_block_lines_get_code_background() {
        let body = "before\n```rust\nlet x = 1;\n```\nafter";
        let lines = render_body(body, &theme());
        // before(0) fence(1) code(2) fence(3) after(4)
        assert_eq!(lines.len(), 5);
        let prose_bg = lines[0].spans[0].style.bg;
        let code_bg = lines[2].spans[0].style.bg;
        assert_eq!(code_bg, Some(Color::Black), "code line carries the code bg");
        assert_ne!(prose_bg, Some(Color::Black), "prose line does not");
    }

    #[test]
    fn trailing_comment_is_dimmed_but_hash_in_string_is_not() {
        let theme = theme();
        // `let s = "a#b"; // note` -> code span + comment span split at '//'
        let lines = render_body("```\nlet s = \"a#b\"; // note\n```", &theme);
        let code = &lines[1];
        assert_eq!(code.spans.len(), 2, "split into code + comment");
        assert!(code.spans[1].content.contains("// note"));
        assert_eq!(code.spans[1].style.fg, Some(Color::DarkGray));
        // The '#' inside the string must NOT have started a comment.
        assert!(code.spans[0].content.contains("a#b"));
    }

    #[test]
    fn inline_code_span_is_tinted_in_prose() {
        let lines = render_body("use the `foo` function", &theme());
        assert_eq!(lines.len(), 1);
        let spans = &lines[0].spans;
        let foo = spans.iter().find(|s| s.content == "foo").expect("inline code span");
        assert_eq!(foo.style.bg, Some(Color::Black));
    }

    #[test]
    fn find_comment_ignores_strings() {
        assert_eq!(find_comment("x = 1 // c"), Some(6));
        assert_eq!(find_comment("x = 1 # c"), Some(6));
        assert_eq!(find_comment("s = \"# not a comment\""), None);
        assert_eq!(find_comment("plain code"), None);
    }
}
