// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, Gauge, List, ListItem, Paragraph, Row, Table, Tabs, Wrap},
    Frame,
};

use crate::{
    app::{App, Tab},
    hipfire::dashboard::{LoadState, VramState},
    hipfire::registry::ModelListItem,
};

const BG: Color = Color::Rgb(7, 7, 9);
const PANEL: Color = Color::Rgb(18, 16, 18);
const PANEL_2: Color = Color::Rgb(40, 24, 27);
const TEXT: Color = Color::Rgb(222, 226, 232);
const MUTED: Color = Color::Rgb(142, 150, 163);
const ACCENT: Color = Color::Rgb(237, 45, 57);
const GREEN: Color = Color::Rgb(102, 217, 139);
const YELLOW: Color = Color::Rgb(238, 190, 95);
const RED: Color = Color::Rgb(255, 95, 104);

/// Tab-bar divider, shared between the renderer (`Tabs::divider`) and the mouse
/// hit-test math in `draw_header` so the two cannot drift (the hit-test advances
/// `x` by exactly this width between tab cells).
const TAB_DIVIDER: &str = " | ";

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    frame.render_widget(Clear, area);
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(10),
            Constraint::Length(2),
        ])
        .split(area);

    draw_header(frame, app, root[0]);
    match app.tab {
        Tab::Home => draw_home(frame, app, root[1]),
        Tab::Dashboard => draw_dashboard(frame, app, root[1]),
        Tab::Chat => draw_chat(frame, app, root[1]),
        Tab::Models => draw_models(frame, app, root[1]),
        Tab::Settings => draw_settings(frame, app, root[1]),
        Tab::System => draw_system(frame, app, root[1]),
        Tab::Logs => draw_logs(frame, app, root[1]),
    }
    draw_footer(frame, app, root[2]);

    // The `?` help overlay draws last, centered on top of everything.
    if app.show_help {
        draw_help_overlay(frame, app, area);
    }
}

/// Centered modal listing the global + active-tab keybindings. Dismissed by any
/// key (handled in main::handle_key). Reuses `footer_hints` for the per-tab line
/// so there is a single source of truth for tab keys.
fn draw_help_overlay(frame: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(
            "Global",
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )),
        Line::from("  Tab / BackTab    switch tabs"),
        Line::from("  ?                toggle this help"),
        Line::from("  r                refresh live data"),
        Line::from("  q                quit"),
        Line::from("  mouse            click a tab · scroll to move in a list"),
        Line::from(""),
        Line::from(Span::styled(
            format!("This tab — {}", app.tab.title()),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )),
    ];
    // The active-tab keys, one per line — keeps each short so nothing wraps and
    // clips inside the modal, and makes the line count (hence height) exact. The
    // globally-listed keys (Tab/BackTab, q) are skipped to avoid duplication.
    for part in footer_hints(app).split('·') {
        let part = part.trim();
        if part.is_empty() || part.starts_with("Tab/BackTab") || part == "q quit" {
            continue;
        }
        lines.push(Line::from(format!("  {part}")));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "press any key to close",
        Style::default().fg(MUTED),
    )));

    let w = 60u16.min(area.width.saturating_sub(2));
    let h = (lines.len() as u16 + 2).min(area.height.saturating_sub(2));
    let x = area.x + area.width.saturating_sub(w) / 2;
    let y = area.y + area.height.saturating_sub(h) / 2;
    let rect = Rect { x, y, width: w, height: h };

    frame.render_widget(Clear, rect);
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(ACCENT))
                    .title(" Keybindings "),
            )
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        rect,
    );
}

fn draw_header(frame: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Length(3)])
        .split(area);

    let title = Line::from(vec![
        Span::styled(
            "hipfire",
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" · control panel", Style::default().fg(ACCENT)),
        Span::styled(
            format!(
                "    serve: {}    model: {}",
                app.status.serve_label(),
                app.active_model
            ),
            Style::default().fg(MUTED),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(title)
            .style(Style::default().bg(BG))
            .alignment(Alignment::Center),
        chunks[0],
    );

    let titles = Tab::ALL
        .iter()
        .map(|tab| Line::from(Span::raw(tab.title())))
        .collect::<Vec<_>>();
    let selected = Tab::ALL.iter().position(|t| *t == app.tab).unwrap_or(0);
    let tabs = Tabs::new(titles)
        .select(selected)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(PANEL_2)),
        )
        .style(Style::default().fg(MUTED).bg(BG))
        .highlight_style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
        .divider(Span::styled(TAB_DIVIDER, Style::default().fg(PANEL_2)));
    frame.render_widget(tabs, chunks[1]);

    // Record per-tab hit regions for mouse clicks. ratatui 0.29's Tabs layout:
    // from the inner left, each tab cell is 1 (left pad) + title + 1 (right pad),
    // with a 3-col " | " divider between cells. The block has only a BOTTOM
    // border, so inner.x == area.x and titles sit on the area's top row.
    app.tab_row_y = chunks[1].y;
    app.tab_hitboxes.clear();
    let right = chunks[1].x.saturating_add(chunks[1].width);
    let mut x = chunks[1].x;
    for tab in Tab::ALL {
        if x >= right {
            break;
        }
        let cell_w = 2 + tab.title().len() as u16; // left pad + title + right pad
        let end = x.saturating_add(cell_w).min(right);
        app.tab_hitboxes.push((x, end, tab));
        x = end.saturating_add(TAB_DIVIDER.len() as u16); // divider between cells
    }
}

/// Per-tab keybind hints shown in the footer. Each tab lists ITS relevant keys;
/// the global `Tab/BackTab switch · q quit` suffix is appended for non-edit
/// states. Kept here (not in App) so the hint text lives next to the renderer.
fn footer_hints(app: &App) -> String {
    let global = "Tab/BackTab switch · q quit";
    match app.tab {
        Tab::Home => format!("r refresh · {global}"),
        Tab::Dashboard => {
            if app.serve_cmd_running() {
                format!("serve {}\u{2026} · {global}", app.serve_cmd_label)
            } else {
                format!("s start · x stop · R restart · r refresh · {global}")
            }
        }
        Tab::Chat => {
            // Chat owns q/r as text input, so it has its own exit hints.
            "Enter send (/help for commands) · Ctrl+O newline · Up/Down scroll · Esc stop / blur".to_string()
        }
        Tab::Models => {
            if app.confirm_delete.is_some() {
                "Delete this model?  y confirm · n / Esc cancel".to_string()
            } else {
                format!("Up/Down select · Enter open/expand · p pull · d delete · r refresh · {global}")
            }
        }
        Tab::Settings => {
            if app.settings_edit.is_some() {
                "Editing: type value · Enter save · Backspace delete · Esc cancel".to_string()
            } else {
                format!("e easy · a advanced · Up/Down select · Left/Right/Space cycle · Enter edit · r refresh · {global}")
            }
        }
        Tab::System => {
            if app.doctor_running() {
                format!("doctor running… · live diagnostics · r refresh · {global}")
            } else {
                format!("d doctor · live diagnostics (auto ~1.5s) · r refresh · {global}")
            }
        }
        Tab::Logs => format!("serve.log tail (auto ~1s) · r refresh · {global}"),
    }
}

fn draw_footer(frame: &mut Frame, app: &App, area: Rect) {
    // A live toast takes over the footer line entirely (colored by severity);
    // otherwise show the per-tab hints + the last persistent status line.
    if let Some(toast) = &app.toast {
        let (fg, tag) = match toast.level {
            crate::app::ToastLevel::Error => (RED, "!"),
            crate::app::ToastLevel::Info => (GREEN, "·"),
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    format!(" {tag} "),
                    Style::default().fg(BG).bg(fg).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" {}", toast.text),
                    Style::default().fg(fg).add_modifier(Modifier::BOLD),
                ),
            ]))
            .style(Style::default().bg(BG)),
            area,
        );
        return;
    }

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(footer_hints(app), Style::default().fg(MUTED)),
            Span::styled(
                format!("    {}", app.last_reload),
                Style::default().fg(Color::DarkGray),
            ),
        ]))
        .style(Style::default().bg(BG)),
        area,
    );
}

fn draw_home(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(pad(area, 1, 0));
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(8)])
        .split(cols[0]);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(12), Constraint::Min(6)])
        .split(cols[1]);

    let serve_color = if app.status.serve_http_ok {
        GREEN
    } else if app.status.serve_pid_alive || app.status.serve_pid.is_some() {
        YELLOW
    } else {
        RED
    };
    let mut status = vec![
        Line::from(vec![
            Span::raw("Serve      "),
            Span::styled(app.status.serve_label(), Style::default().fg(serve_color)),
        ]),
        Line::from(format!(
            "Endpoint   {}:{}",
            app.config.probe_host(),
            app.config.port
        )),
        Line::from(format!(
            "PID        {}",
            app.status
                .serve_pid
                .map(|p| p.to_string())
                .unwrap_or_else(|| "-".into())
        )),
        Line::from(format!("Active     {}", app.active_model)),
        Line::from(format!(
            "Config     {} ({})",
            app.config.default_model,
            if app.config.loaded_from_disk {
                "custom"
            } else {
                "defaults"
            }
        )),
        Line::from(format!(
            "Overrides  {} model overlays",
            app.config.per_model_count
        )),
        Line::from(format!(
            "Models     {} local / {} registry",
            app.registry.local_files.len(),
            app.registry.models.len()
        )),
    ];
    if let Some(warning) = &app.config.warning {
        status.push(Line::from(vec![
            Span::styled("Config     ", Style::default().fg(YELLOW)),
            Span::styled(warning.clone(), Style::default().fg(YELLOW)),
        ]));
    }
    if let Some(warning) = &app.registry.warning {
        status.push(Line::from(vec![
            Span::styled("Registry   ", Style::default().fg(YELLOW)),
            Span::styled(warning.clone(), Style::default().fg(YELLOW)),
        ]));
    }
    frame.render_widget(card("Runtime", status), left[0]);

    let actions = vec![
        ListItem::new("Dashboard — live serve status, queue depth, tok/s, VRAM."),
        ListItem::new("Chat — stream a conversation through your local serve."),
        ListItem::new("Models — browse the registry and your downloads."),
        ListItem::new("Settings — edit config (easy/advanced); persists to ~/.hipfire."),
        ListItem::new("System — GPU, HIP, kernel cache, and path checks."),
    ];
    frame.render_widget(
        List::new(actions)
            .block(block("Tabs  (Tab / BackTab to switch)"))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        left[1],
    );

    let quickstart = Text::from(vec![
        Line::from(Span::styled(
            "Quick start",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from("1. Pull a model   ·  hipfire pull qwen3.5:4b"),
        Line::from("2. Start serving  ·  hipfire serve -d"),
        Line::from("3. Chat           ·  open the Chat tab and type"),
        Line::from(""),
        Line::from(Span::styled(
            "Everything here reads your live local state and the",
            Style::default().fg(MUTED),
        )),
        Line::from(Span::styled(
            "running daemon — edits persist to ~/.hipfire.",
            Style::default().fg(MUTED),
        )),
    ]);
    frame.render_widget(
        Paragraph::new(quickstart)
            .block(block("Getting started"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        right[0],
    );

    let keys = vec![
        Row::new(["Tab / BackTab", "switch tabs"]),
        Row::new(["\u{2191}  \u{2193}", "move within a tab"]),
        Row::new(["Enter", "select / send"]),
        Row::new(["r", "refresh live data"]),
        Row::new(["q", "quit"]),
    ];
    frame.render_widget(
        Table::new(
            keys,
            [Constraint::Length(16), Constraint::Min(20)],
        )
        .header(Row::new(["Key", "Action"]).style(Style::default().fg(MUTED)))
        .block(block("Keys  (full list in each tab's footer)"))
        .style(Style::default().fg(TEXT).bg(PANEL))
        .row_highlight_style(Style::default().bg(PANEL_2)),
        right[1],
    );
}

fn draw_dashboard(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(pad(area, 1, 0));

    // ── Serve panel ─────────────────────────────────────────────────────
    let mut serve_lines: Vec<Line> = Vec::new();
    let dash = app.dashboard.as_ref();
    let serve_up = dash.map(|d| d.serve_up).unwrap_or(false);

    if dash.is_none() {
        serve_lines.push(Line::from(Span::styled(
            "Probing serve…",
            Style::default().fg(MUTED),
        )));
    } else if !serve_up {
        let d = dash.unwrap();
        serve_lines.push(Line::from(vec![
            Span::raw("Serve      "),
            Span::styled("offline", Style::default().fg(RED).add_modifier(Modifier::BOLD)),
        ]));
        serve_lines.push(Line::from(format!("Endpoint   {}", d.endpoint)));
        serve_lines.push(Line::from(""));
        serve_lines.push(Line::from(Span::styled(
            d.offline_hint
                .clone()
                .unwrap_or_else(|| "serve offline — start it with `hipfire serve -d`".into()),
            Style::default().fg(YELLOW),
        )));
        serve_lines.push(Line::from(Span::styled(
            "(Chat tab Enter also starts a background serve)",
            Style::default().fg(MUTED),
        )));
    } else {
        let d = dash.unwrap();
        serve_lines.push(Line::from(vec![
            Span::raw("Serve      "),
            Span::styled("online", Style::default().fg(GREEN).add_modifier(Modifier::BOLD)),
        ]));
        serve_lines.push(Line::from(format!("Endpoint   {}", d.endpoint)));
        serve_lines.push(Line::from(format!(
            "Model      {}",
            d.model.clone().unwrap_or_else(|| "(none loaded)".into())
        )));
        if let Some(s) = &d.stats {
            serve_lines.push(Line::from(format!("Uptime     {}", fmt_uptime(s.uptime_s))));
            serve_lines.push(Line::from(vec![
                Span::raw("Queue      "),
                Span::styled(
                    s.queue_depth.to_string(),
                    Style::default().fg(if s.queue_depth > 0 { YELLOW } else { TEXT }),
                ),
                Span::raw("  in-flight"),
            ]));
            serve_lines.push(Line::from(format!("Requests   {} served", s.requests_served)));
            serve_lines.push(Line::from(match s.recent_tok_s {
                Some(t) => format!("Recent     {t:.1} tok/s"),
                None => "Recent     — (no completed generation yet)".into(),
            }));
        } else {
            serve_lines.push(Line::from(Span::styled(
                "stats unavailable (older serve build, /stats not present)",
                Style::default().fg(YELLOW),
            )));
        }
        if !d.model_ids.is_empty() {
            serve_lines.push(Line::from(format!(
                "Available  {} model(s)",
                d.model_ids.len()
            )));
        }
    }
    frame.render_widget(card("Live serve", serve_lines), cols[0]);

    // ── VRAM panel ──────────────────────────────────────────────────────
    let mut vram_lines: Vec<Line> = Vec::new();
    match dash.map(|d| &d.vram) {
        Some(VramState::Available(gpus)) => {
            for g in gpus {
                let pct = if g.total_bytes > 0 {
                    (g.used_bytes as f64 / g.total_bytes as f64) * 100.0
                } else {
                    0.0
                };
                let color = if pct > 90.0 {
                    RED
                } else if pct > 70.0 {
                    YELLOW
                } else {
                    GREEN
                };
                vram_lines.push(Line::from(vec![
                    Span::styled(
                        format!("GPU {}  ", g.index),
                        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(
                            "{:.1} / {:.1} GB used ({pct:.0}%)",
                            g.used_gb(),
                            g.total_gb()
                        ),
                        Style::default().fg(color),
                    ),
                ]));
                vram_lines.push(Line::from(Span::styled(
                    format!("        {:.1} GB free", g.free_gb()),
                    Style::default().fg(MUTED),
                )));
            }
        }
        Some(VramState::Unavailable(reason)) => {
            vram_lines.push(Line::from(Span::styled(
                reason.clone(),
                Style::default().fg(YELLOW),
            )));
        }
        None => {
            vram_lines.push(Line::from(Span::styled(
                "Probing rocm-smi…",
                Style::default().fg(MUTED),
            )));
        }
    }

    // ── Live load (util / temp / power) ─────────────────────────────────
    match dash.map(|d| &d.gpu_load) {
        Some(LoadState::Available(loads)) => {
            vram_lines.push(Line::from(""));
            for l in loads {
                let mut parts = Vec::new();
                if let Some(u) = l.util_pct {
                    parts.push(format!("{u:.0}% util"));
                }
                if let Some(t) = l.temp_c {
                    parts.push(format!("{t:.0}°C"));
                }
                if let Some(p) = l.power_w {
                    parts.push(format!("{p:.0} W"));
                }
                let color = match l.util_pct {
                    Some(u) if u > 90.0 => RED,
                    Some(u) if u > 50.0 => YELLOW,
                    _ => GREEN,
                };
                let body = if parts.is_empty() {
                    "no load fields".to_string()
                } else {
                    parts.join("  ·  ")
                };
                vram_lines.push(Line::from(vec![
                    Span::styled(
                        format!("GPU {}  ", l.index),
                        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(body, Style::default().fg(color)),
                ]));
            }
        }
        Some(LoadState::Unavailable(reason)) => {
            vram_lines.push(Line::from(""));
            vram_lines.push(Line::from(Span::styled(
                reason.clone(),
                Style::default().fg(MUTED),
            )));
        }
        None => {}
    }
    frame.render_widget(card("GPU (rocm-smi)", vram_lines), cols[1]);
}

fn fmt_uptime(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h}h {m}m {s}s")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

fn draw_chat(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(5)])
        .split(pad(area, 1, 0));

    let mut lines = Vec::new();
    if app.chat.messages.is_empty() {
        lines.push(Line::from(Span::styled(
            "No messages yet. Type below and press Enter.",
            Style::default().fg(MUTED),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(
            "Chat streams from your local hipfire serve (OpenAI-compatible).",
        ));
    } else {
        // Fenced ``` code blocks get a distinct background; comments + inline
        // `code` are tinted. Rendering is isolated in `ui_chat`.
        let code_theme = crate::ui_chat::CodeTheme {
            text: TEXT,
            code_fg: ACCENT,
            code_bg: PANEL_2,
            comment: MUTED,
            fence: YELLOW,
        };
        for msg in &app.chat.messages {
            let color = if msg.role == "user" { ACCENT } else { GREEN };
            lines.push(Line::from(Span::styled(
                format!("{}:", msg.role),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            )));
            lines.extend(crate::ui_chat::render_body(&msg.content, &code_theme));
            lines.push(Line::from(""));
        }
    }
    frame.render_widget(
        Paragraph::new(lines)
            .block(block("Chat shell"))
            .scroll((app.chat.scroll, 0))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        chunks[0],
    );

    let input_title = if app.chat.sending {
        format!("Input - {} - model {}", app.chat.status, app.active_model)
    } else {
        let mut extras = vec![format!("model {}", app.active_model)];
        if !app.chat.system_prompt.is_empty() {
            extras.push("sys".into());
        }
        if let Some(t) = app.chat.temp {
            extras.push(format!("temp {t}"));
        }
        if let Some(p) = app.chat.top_p {
            extras.push(format!("top_p {p}"));
        }
        if let Some(s) = app.chat.last_stats {
            extras.push(format!("{} tok @ {:.0} tok/s", s.tokens, s.tps));
        }
        format!("Input - {} - {}", app.chat.status, extras.join(" · "))
    };
    let input = Paragraph::new(app.chat.input.as_str())
        .block(block(&input_title))
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(TEXT).bg(PANEL_2));
    frame.render_widget(input, chunks[1]);
}

fn draw_models(frame: &mut Frame, app: &App, area: Rect) {
    // An in-flight pull or an armed delete-confirm gets a 3-row status strip at
    // the bottom; otherwise the list takes the full height.
    let show_status = app.pull.is_some() || app.confirm_delete.is_some();
    let constraints: Vec<Constraint> = if show_status {
        vec![
            Constraint::Length(3),
            Constraint::Min(6),
            Constraint::Length(3),
        ]
    } else {
        vec![Constraint::Length(3), Constraint::Min(8)]
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(pad(area, 1, 0));
    let summary = format!(
        "active: {}    {} downloaded / {} available    registry: {}    aliases: {}",
        app.active_model,
        app.registry.downloaded_count(),
        app.registry.models.len(),
        app.registry
            .loaded_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "missing".into()),
        app.registry.aliases.len(),
    );
    frame.render_widget(
        Paragraph::new(summary)
            .block(block("Model hub"))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        chunks[0],
    );

    let visible_items = app.registry.visible_items();
    if visible_items.is_empty() {
        // Empty-state guidance: no registry entries and no local downloads.
        let lines = vec![
            Line::from(Span::styled(
                "No models found.",
                Style::default().fg(YELLOW).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("Pull a model with:"),
            Line::from(Span::styled(
                "    hipfire pull <id>",
                Style::default().fg(GREEN),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "The registry (cli/registry.json) lists available ids; downloads land in ~/.hipfire/models.",
                Style::default().fg(MUTED),
            )),
        ];
        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .block(block("Registry browser"))
                .wrap(Wrap { trim: false })
                .style(Style::default().fg(TEXT).bg(PANEL)),
            chunks[1],
        );
        return;
    }
    let rows = visible_items
        .iter()
        .enumerate()
        .skip(scroll_start(app.registry.selected, chunks[1].height, 3))
        .take(visible_rows(chunks[1].height, 3))
        .map(|(idx, item)| {
            let selected = idx == app.registry.selected;
            let row = match item {
                ModelListItem::Group {
                    name,
                    count,
                    downloaded,
                    expanded,
                } => {
                    let marker = if *expanded { "v" } else { ">" };
                    Row::new([
                        format!("{marker} {name}"),
                        format!("{downloaded}/{count} local"),
                        String::new(),
                        String::new(),
                        String::new(),
                        "Enter/Right to expand, Left to collapse".into(),
                    ])
                }
                ModelListItem::Model { model_index } => {
                    let row = &app.registry.models[*model_index];
                    let status = if row.tag == app.active_model {
                        "active"
                    } else if row.downloaded {
                        "local"
                    } else if row.entry.repo.is_empty() {
                        "local-only"
                    } else {
                        "remote"
                    };
                    let extras = match (row.has_triattn, row.has_mtp) {
                        (true, true) => "triattn mtp",
                        (true, false) => "triattn",
                        (false, true) => "mtp",
                        _ => "",
                    };
                    Row::new([
                        format!("  {}", row.tag),
                        status.into(),
                        format!("{:.1} GB", row.entry.size_gb),
                        format!("{:.0} GB", row.entry.min_vram_gb),
                        extras.into(),
                        if row.entry.repo.is_empty() {
                            format!("{} (no remote repo)", row.entry.desc)
                        } else {
                            row.entry.desc.clone()
                        },
                    ])
                }
            };
            row.style(if selected {
                Style::default().fg(ACCENT).bg(PANEL_2)
            } else {
                match item {
                    ModelListItem::Group { .. } => Style::default()
                        .fg(YELLOW)
                        .bg(PANEL)
                        .add_modifier(Modifier::BOLD),
                    ModelListItem::Model { model_index } => {
                        if app.registry.models[*model_index].tag == app.active_model {
                            Style::default().fg(GREEN).bg(PANEL)
                        } else {
                            Style::default().fg(TEXT).bg(PANEL)
                        }
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    let table = Table::new(
        rows,
        [
            Constraint::Length(24),
            Constraint::Length(8),
            Constraint::Length(9),
            Constraint::Length(8),
            Constraint::Length(12),
            Constraint::Min(20),
        ],
    )
    .header(
        Row::new(["Tag", "Have", "Size", "VRAM", "Sidecars", "Notes"])
            .style(Style::default().fg(MUTED)),
    )
    .block(block("Registry browser"))
    .style(Style::default().bg(PANEL));
    frame.render_widget(table, chunks[1]);

    if show_status {
        draw_models_status(frame, app, chunks[2]);
    }
}

/// The Models-tab bottom strip: a delete-confirm prompt (takes priority) or the
/// in-flight pull's progress gauge (parsed percent + the raw CLI line label).
fn draw_models_status(frame: &mut Frame, app: &App, area: Rect) {
    if let Some(tag) = &app.confirm_delete {
        frame.render_widget(
            Paragraph::new(format!("Delete {tag}?  press y to confirm, n / Esc to cancel"))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(RED))
                        .title(" Confirm delete "),
                )
                .style(Style::default().fg(TEXT).bg(PANEL)),
            area,
        );
    } else if let Some(job) = &app.pull {
        let ratio = job.percent.unwrap_or(0.0).clamp(0.0, 100.0) / 100.0;
        let label = if job.line.is_empty() {
            format!("pulling {}\u{2026}", job.tag)
        } else {
            job.line.clone()
        };
        frame.render_widget(
            Gauge::default()
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(format!(" Pulling {} ", job.tag)),
                )
                .gauge_style(Style::default().fg(GREEN).bg(PANEL_2))
                .ratio(ratio)
                .label(label),
            area,
        );
    }
}

fn draw_settings(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(8)])
        .split(pad(area, 1, 0));
    let mode = if app.settings_easy {
        "Easy settings"
    } else {
        "Advanced settings"
    };
    let note = if let Some(edit) = &app.settings_edit {
        // Show the live edit buffer.
        format!("editing {} = {}_  (Enter save, Esc cancel)", edit.key, edit.buffer)
    } else if app.settings_easy {
        "Writable. Left/Right cycle, Enter edits numbers. Press a for advanced.".to_string()
    } else {
        "Writable raw config. Left/Right cycle, Enter edits. Press e for easy.".to_string()
    };
    frame.render_widget(
        Paragraph::new(format!("{mode}    {note}"))
            .block(block("Settings"))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        chunks[0],
    );

    if app.settings_easy {
        let rows_all = app
            .config
            .easy_rows()
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>();
        let start = scroll_start(app.settings_selected, chunks[1].height, 3);
        let rows = rows_all
            .into_iter()
            .skip(start)
            .take(visible_rows(chunks[1].height, 3))
            .map(|(idx, (label, value, desc))| {
                Row::new([label.to_string(), value, desc.to_string()]).style(
                    if idx == app.settings_selected {
                        Style::default().fg(ACCENT).bg(PANEL_2)
                    } else {
                        Style::default().fg(TEXT).bg(PANEL)
                    },
                )
            })
            .collect::<Vec<_>>();
        frame.render_widget(
            Table::new(
                rows,
                [
                    Constraint::Length(16),
                    Constraint::Length(24),
                    Constraint::Min(30),
                ],
            )
            .header(Row::new(["Setting", "Value", "Meaning"]).style(Style::default().fg(MUTED)))
            .block(block("User-safe controls"))
            .style(Style::default().fg(TEXT).bg(PANEL)),
            chunks[1],
        );
    } else {
        let rows_all = app.config.values.iter().enumerate().collect::<Vec<_>>();
        let start = scroll_start(app.settings_selected, chunks[1].height, 3);
        let rows = rows_all
            .into_iter()
            .skip(start)
            .take(visible_rows(chunks[1].height, 3))
            .map(|(idx, (k, v))| {
                Row::new([k.clone(), v.clone()]).style(if idx == app.settings_selected {
                    Style::default().fg(ACCENT).bg(PANEL_2)
                } else {
                    Style::default().fg(TEXT).bg(PANEL)
                })
            })
            .collect::<Vec<_>>();
        frame.render_widget(
            Table::new(rows, [Constraint::Length(28), Constraint::Min(20)])
                .header(Row::new(["Key", "Value"]).style(Style::default().fg(MUTED)))
                .block(block("Advanced config.json view"))
                .style(Style::default().fg(TEXT).bg(PANEL)),
            chunks[1],
        );
    }
}

fn draw_system(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .split(pad(area, 1, 0));
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(8)])
        .split(cols[0]);

    let gpu_lines = app
        .status
        .gpu_lines
        .iter()
        .map(|line| Line::from(line.clone()))
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(gpu_lines)
            .block(block("Hardware glimpse"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        left[0],
    );

    let paths = app
        .status
        .paths_ok
        .iter()
        .map(|(label, ok)| {
            Row::new([
                label.clone(),
                if *ok {
                    "present".into()
                } else {
                    "missing".into()
                },
            ])
            .style(if *ok {
                Style::default().fg(GREEN)
            } else {
                Style::default().fg(YELLOW)
            })
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Table::new(paths, [Constraint::Length(24), Constraint::Length(12)])
            .header(Row::new(["Path", "Status"]).style(Style::default().fg(MUTED)))
            .block(block("Files"))
            .style(Style::default().bg(PANEL)),
        left[1],
    );

    let mut diagnostic_lines = vec![Line::from(Span::styled(
        "Live diagnostics",
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    ))];
    match app.dashboard.as_ref().and_then(|d| d.system.as_ref()) {
        Some(sys) => {
            diagnostic_lines.push(probe_line("GPU", &sys.gpu_name));
            diagnostic_lines.push(probe_line("Arch", &sys.gpu_arch));
            diagnostic_lines.push(probe_line("HIP/ROCm", &sys.hip_version));
            diagnostic_lines.push(probe_line("Kernel cache", &sys.kernel_cache));
            diagnostic_lines.push(probe_line("Loaded model", &sys.loaded_model));
            diagnostic_lines.push(probe_line("Checksum", &sys.model_checksum));
        }
        None => {
            // Worker has not produced a snapshot yet (first frame on this tab).
            diagnostic_lines.push(Line::from(Span::styled(
                "Probing GPU / HIP / kernel cache…",
                Style::default().fg(MUTED),
            )));
        }
    }
    diagnostic_lines.extend([
        Line::from(""),
        Line::from(Span::styled(
            "Local model files:",
            Style::default().fg(MUTED),
        )),
    ]);
    if app.registry.local_files.is_empty() {
        // Empty-state guidance: tell the user how to get a model.
        diagnostic_lines.push(Line::from(Span::styled(
            "No local models under ~/.hipfire/models.",
            Style::default().fg(YELLOW),
        )));
        diagnostic_lines.push(Line::from(Span::styled(
            "Pull one with `hipfire pull <id>` (see the Models tab for ids).",
            Style::default().fg(YELLOW),
        )));
    } else {
        diagnostic_lines.extend(
            app.registry
                .local_files
                .iter()
                .take(6)
                .map(|m| Line::from(format!("{}  {}", m.size, m.file))),
        );
    }
    // Request inspector — the TUI's own recent chat generations (honest, local).
    diagnostic_lines.push(Line::from(""));
    diagnostic_lines.push(Line::from(Span::styled(
        "Recent requests (this session):",
        Style::default().fg(MUTED),
    )));
    if app.request_log.is_empty() {
        diagnostic_lines.push(Line::from(Span::styled(
            "none yet — send a Chat message",
            Style::default().fg(MUTED),
        )));
    } else {
        for r in app.request_log.iter().take(8) {
            diagnostic_lines.push(Line::from(format!(
                "{}  {} tok · {:.0} tok/s · {:.1}s",
                r.model, r.tokens, r.tps, r.secs
            )));
        }
    }

    // In-TUI doctor — `hipfire diag` pass/fail (on-demand via `d`).
    diagnostic_lines.push(Line::from(""));
    if app.doctor_running() {
        diagnostic_lines.push(Line::from(Span::styled(
            "Doctor: running hipfire diag…",
            Style::default().fg(YELLOW),
        )));
    } else {
        match app.doctor.as_ref() {
            None => diagnostic_lines.push(Line::from(Span::styled(
                "Doctor: press d to run hipfire diag",
                Style::default().fg(MUTED),
            ))),
            Some(report) if report.error.is_some() => {
                diagnostic_lines.push(Line::from(Span::styled(
                    format!("Doctor error: {}", report.error.as_deref().unwrap_or("")),
                    Style::default().fg(RED),
                )));
            }
            Some(report) => {
                diagnostic_lines.push(Line::from(Span::styled(
                    "Doctor (hipfire diag):",
                    Style::default().fg(MUTED),
                )));
                for c in &report.checks {
                    let (mark, color) = if c.ok { ("ok ", GREEN) } else { ("XX ", RED) };
                    diagnostic_lines.push(Line::from(vec![
                        Span::styled(format!("  {mark}"), Style::default().fg(color)),
                        Span::styled(format!("{}: ", c.name), Style::default().fg(TEXT)),
                        Span::styled(c.detail.clone(), Style::default().fg(MUTED)),
                    ]));
                }
            }
        }
    }

    diagnostic_lines.extend([
        Line::from(""),
        Line::from(Span::styled(
            "Serve /health response:",
            Style::default().fg(MUTED),
        )),
        Line::from(if app.status.health_text.is_empty() {
            "serve offline — start with `hipfire serve -d`".to_string()
        } else {
            app.status.health_text.chars().take(300).collect::<String>()
        }),
    ]);
    frame.render_widget(
        Paragraph::new(Text::from(diagnostic_lines))
            .block(block("System"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        cols[1],
    );
}

/// Render one `label: value` diagnostic line, coloring an honest "unavailable"
/// probe yellow and a real value green-tinted text.
fn probe_line(label: &str, probe: &crate::hipfire::dashboard::Probe) -> Line<'static> {
    let value_style = if probe.is_available() {
        Style::default().fg(TEXT)
    } else {
        Style::default().fg(YELLOW)
    };
    Line::from(vec![
        Span::styled(format!("{label:<13}"), Style::default().fg(MUTED)),
        Span::styled(probe.display().to_string(), value_style),
    ])
}

fn card(title: &str, lines: Vec<Line<'static>>) -> Paragraph<'static> {
    Paragraph::new(lines)
        .block(block(title))
        .style(Style::default().fg(TEXT).bg(PANEL))
        .wrap(Wrap { trim: false })
}

fn draw_logs(frame: &mut Frame, app: &App, area: Rect) {
    use crate::hipfire::log_tail::LogStatus;
    let inner = pad(area, 1, 0);
    let (title, body): (String, Text) = match &app.logs.status {
        LogStatus::Pending => (
            "serve.log".into(),
            Text::from(Line::from(Span::styled(
                "reading serve.log\u{2026}",
                Style::default().fg(MUTED),
            ))),
        ),
        LogStatus::Missing => (
            "serve.log (not found)".into(),
            Text::from(vec![
                Line::from(Span::styled("No serve.log yet.", Style::default().fg(YELLOW))),
                Line::from(""),
                Line::from(Span::styled(
                    "Start serve (Dashboard tab: s) — its output lands in ~/.hipfire/serve.log.",
                    Style::default().fg(MUTED),
                )),
            ]),
        ),
        LogStatus::Empty => (
            "serve.log (empty)".into(),
            Text::from(Line::from(Span::styled(
                "serve.log is empty.",
                Style::default().fg(MUTED),
            ))),
        ),
        LogStatus::Error(e) => (
            "serve.log (error)".into(),
            Text::from(Line::from(Span::styled(
                format!("read error: {e}"),
                Style::default().fg(RED),
            ))),
        ),
        LogStatus::Ok => {
            let title = format!("serve.log (last {} lines)", app.logs.lines.len());
            // Show the tail that fits, newest at the bottom.
            let height = inner.height.saturating_sub(2) as usize; // borders
            let start = app.logs.lines.len().saturating_sub(height.max(1));
            let lines: Vec<Line> = app.logs.lines[start..]
                .iter()
                .map(|l| Line::from(l.as_str()))
                .collect();
            (title, Text::from(lines))
        }
    };
    frame.render_widget(
        Paragraph::new(body)
            .block(block(&title))
            .style(Style::default().fg(TEXT).bg(PANEL)),
        inner,
    );
}

fn block(title: &str) -> Block<'static> {
    Block::default()
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(PANEL_2))
        .style(Style::default().bg(PANEL))
}

fn pad(area: Rect, x: u16, y: u16) -> Rect {
    Rect {
        x: area.x.saturating_add(x),
        y: area.y.saturating_add(y),
        width: area.width.saturating_sub(x * 2),
        height: area.height.saturating_sub(y * 2),
    }
}

fn visible_rows(height: u16, chrome: u16) -> usize {
    height.saturating_sub(chrome).max(1) as usize
}

fn scroll_start(selected: usize, height: u16, chrome: u16) -> usize {
    let visible = visible_rows(height, chrome);
    selected.saturating_sub(visible.saturating_sub(1))
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::app::App;
    use crate::hipfire::dashboard::{Dashboard, GpuLoad, GpuMem, LoadState, ServeStats, VramState};
    use ratatui::{backend::TestBackend, Terminal};

    /// Render the whole UI for the Dashboard tab with the given snapshot and
    /// return the flattened buffer text. App::load() reads ~/.hipfire but is
    /// tolerant of missing files (same path as `--check`); the Dashboard panel
    /// itself draws purely from `app.dashboard`, which we inject here.
    fn render_dashboard(dash: Option<Dashboard>) -> String {
        let mut app = App::load().expect("App::load");
        app.tab = Tab::Dashboard;
        app.dashboard = dash;
        let backend = TestBackend::new(110, 30);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("draw must not panic");
        let buf = terminal.backend().buffer().clone();
        buf.content().iter().map(|c| c.symbol()).collect()
    }

    #[test]
    fn renders_online_with_stats_and_vram() {
        let dash = Dashboard {
            serve_up: true,
            endpoint: "127.0.0.1:11435".into(),
            health_text: r#"{"status":"ok","model":"qwen3.5:9b"}"#.into(),
            model: Some("qwen3.5:9b".into()),
            stats: Some(ServeStats {
                model: Some("qwen3.5:9b".into()),
                uptime_s: 3725, // 1h 2m 5s
                queue_depth: 1,
                requests_served: 7,
                recent_tok_s: Some(151.2),
            }),
            model_ids: vec!["qwen3.5:9b".into(), "qwen3.5:27b".into()],
            vram: VramState::Available(vec![GpuMem {
                index: 0,
                used_bytes: 8_000_000_000,
                total_bytes: 24_000_000_000,
            }]),
            gpu_load: LoadState::Available(vec![GpuLoad {
                index: 0,
                util_pct: Some(37.0),
                temp_c: Some(45.0),
                power_w: Some(120.0),
            }]),
            offline_hint: None,
            system: None,
        };
        let text = render_dashboard(Some(dash));
        assert!(text.contains("online"), "expected online marker");
        assert!(text.contains("qwen3.5:9b"), "expected model name");
        assert!(text.contains("127.0.0.1:11435"), "expected endpoint");
        assert!(text.contains("1h 2m 5s"), "expected uptime");
        assert!(text.contains("served"), "expected requests served");
        assert!(text.contains("tok/s"), "expected recent tok/s");
        assert!(text.contains("GPU 0"), "expected per-GPU VRAM");
        assert!(text.contains("37% util"), "expected live GPU utilization");
        assert!(text.contains("120 W"), "expected live GPU power");
    }

    #[test]
    fn renders_offline_state_without_panic_or_stale_numbers() {
        let dash = Dashboard::offline(
            "127.0.0.1:11435".into(),
            VramState::Unavailable("VRAM unavailable: rocm-smi not installed".into()),
            LoadState::Unavailable("GPU load unavailable: rocm-smi not installed".into()),
        );
        let text = render_dashboard(Some(dash));
        assert!(text.contains("offline"), "expected offline marker");
        assert!(text.contains("hipfire serve"), "expected start hint");
        assert!(
            text.contains("VRAM unavailable"),
            "expected honest VRAM-unavailable line"
        );
        // No fabricated telemetry: offline must not invent a tok/s or queue.
        assert!(!text.contains("tok/s"), "offline must not show tok/s");
    }

    #[test]
    fn renders_before_first_probe() {
        // dashboard == None (initial frame) must render honest "probing" text,
        // not a crash and not zeros-as-data.
        let text = render_dashboard(None);
        assert!(text.contains("Probing serve"), "expected probing placeholder");
    }

    use crate::hipfire::dashboard::{Probe, SystemInfo};

    /// Render any tab and return flattened buffer text. Mutates `app` via the
    /// passed closure first (set tab / inject dashboard / toast). CPU-only, no
    /// TTY, no GPU, no serve.
    fn render_with(setup: impl FnOnce(&mut App)) -> String {
        let mut app = App::load().expect("App::load");
        setup(&mut app);
        let backend = TestBackend::new(120, 32);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("draw must not panic");
        let buf = terminal.backend().buffer().clone();
        buf.content().iter().map(|c| c.symbol()).collect()
    }

    fn dash_with_system(system: SystemInfo) -> Dashboard {
        let mut d = Dashboard::offline(
            "127.0.0.1:11435".into(),
            VramState::Unavailable("VRAM unavailable: rocm-smi not installed".into()),
            LoadState::Unavailable("GPU load unavailable: rocm-smi not installed".into()),
        );
        d.system = Some(system);
        d
    }

    #[test]
    fn system_tab_renders_live_diagnostics() {
        let sys = SystemInfo {
            gpu_name: Probe::Value("Radeon RX 7900 XTX".into()),
            gpu_arch: Probe::Value("gfx1100".into()),
            hip_version: Probe::Value("HIP 6.2.41134".into()),
            kernel_cache: Probe::Value("/home/u/.hipfire_kernels (present, 12 entries)".into()),
            loaded_model: Probe::Value("/home/u/.hipfire/models/q.mq4".into()),
            model_checksum: Probe::Value("deadbeefdeadbeef (4.20 GB)".into()),
        };
        let text = render_with(|app| {
            app.tab = Tab::System;
            app.dashboard = Some(dash_with_system(sys.clone()));
        });
        assert!(text.contains("Live diagnostics"), "expected live header");
        assert!(text.contains("Radeon RX 7900 XTX"), "expected gpu name");
        assert!(text.contains("gfx1100"), "expected gpu arch");
        assert!(text.contains("HIP 6.2"), "expected HIP version");
        assert!(text.contains(".hipfire_kernels"), "expected kernel cache path");
        assert!(text.contains("deadbeef"), "expected model checksum");
        // Must NOT show the old static placeholder.
        assert!(
            !text.contains("Diagnostics roadmap"),
            "System tab must be live, not the static placeholder"
        );
    }

    #[test]
    fn system_tab_renders_unavailable_states_honestly() {
        let sys = SystemInfo {
            gpu_name: Probe::Unavailable("unavailable: rocm-smi not installed".into()),
            gpu_arch: Probe::Unavailable("unavailable: rocminfo not installed".into()),
            hip_version: Probe::Unavailable("unavailable: hipconfig absent".into()),
            kernel_cache: Probe::Unavailable("/home/u/.hipfire_kernels (absent — populated on first run)".into()),
            loaded_model: Probe::Unavailable("no model loaded".into()),
            model_checksum: Probe::Unavailable("no model loaded".into()),
        };
        let text = render_with(|app| {
            app.tab = Tab::System;
            app.dashboard = Some(dash_with_system(sys));
        });
        assert!(text.contains("unavailable"), "expected honest unavailable");
        assert!(text.contains("no model loaded"), "expected idle model state");
        // No fabricated gfx string when the probe failed.
        assert!(!text.contains("gfx1"), "must not invent a gfx target");
    }

    #[test]
    fn system_tab_before_first_probe_shows_probing() {
        let text = render_with(|app| {
            app.tab = Tab::System;
            app.dashboard = None;
        });
        assert!(
            text.contains("Probing GPU"),
            "expected probing placeholder before first snapshot"
        );
    }

    #[test]
    fn footer_hints_are_per_tab() {
        let settings = render_with(|app| app.tab = Tab::Settings);
        assert!(settings.contains("cycle"), "settings footer mentions cycle");
        assert!(settings.contains("easy"), "settings footer mentions easy");

        let models = render_with(|app| app.tab = Tab::Models);
        assert!(models.contains("select"), "models footer mentions select");
        assert!(models.contains("expand"), "models footer mentions expand");

        let dash = render_with(|app| app.tab = Tab::Dashboard);
        assert!(dash.contains("refresh"), "dashboard footer mentions refresh");

        let chat = render_with(|app| app.tab = Tab::Chat);
        assert!(chat.contains("send"), "chat footer mentions send");

        let system = render_with(|app| app.tab = Tab::System);
        assert!(system.contains("diagnostics"), "system footer mentions diagnostics");
        // Global hints present on a non-chat tab.
        assert!(dash.contains("quit"), "global quit hint present");
    }

    #[test]
    fn error_toast_overrides_footer() {
        let text = render_with(|app| {
            app.tab = Tab::Settings;
            app.toast_error("save failed: permission denied");
        });
        assert!(
            text.contains("save failed: permission denied"),
            "expected error toast text in footer"
        );
    }

    #[test]
    fn models_empty_state_guides_to_pull() {
        // App::load with no registry / no local models yields empty visible
        // items; the Models tab must render pull guidance, not an empty table.
        let text = render_with(|app| {
            app.tab = Tab::Models;
            app.registry.models.clear();
            app.registry.local_files.clear();
        });
        assert!(text.contains("No models found"), "expected empty-state header");
        assert!(text.contains("hipfire pull"), "expected pull guidance");
    }

    #[test]
    fn help_overlay_renders_keybindings() {
        let text = render_with(|app| app.show_help = true);
        assert!(text.contains("Keybindings"), "help overlay title");
        assert!(text.contains("switch tabs"), "global keys listed");
        assert!(text.contains("toggle this help"), "the ? key is documented");
    }

    #[test]
    fn header_records_ordered_tab_hitboxes() {
        let mut app = App::load().expect("App::load");
        let backend = TestBackend::new(110, 30);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("draw must not panic");
        assert_eq!(app.tab_hitboxes.len(), Tab::ALL.len());
        for (i, (start, end, tab)) in app.tab_hitboxes.iter().enumerate() {
            assert!(end > start, "tab {tab:?} has an empty hit region");
            assert_eq!(*tab, Tab::ALL[i], "hit regions ordered like Tab::ALL");
        }
        // A click inside the first tab's region resolves to Home.
        let (start, _, _) = app.tab_hitboxes[0];
        assert_eq!(app.tab_at(start, app.tab_row_y), Some(Tab::Home));
    }

    #[test]
    fn tab_hitboxes_align_with_rendered_columns() {
        // Cross-check each recorded hit region against where ratatui ACTUALLY
        // drew the tab title in the buffer — the only check that catches a real
        // off-by-one in draw_header's cell-width / divider math (the fabricated-
        // hitbox click tests cannot).
        let mut app = App::load().expect("App::load");
        let backend = TestBackend::new(110, 30);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("draw must not panic");
        let buf = terminal.backend().buffer().clone();
        let width = buf.area().width as usize;
        let row = app.tab_row_y as usize;
        let content = buf.content();
        let row_text: String = (0..width).map(|c| content[row * width + c].symbol()).collect();
        for (start, _end, tab) in &app.tab_hitboxes {
            let title = tab.title();
            let title_col = (*start as usize) + 1; // title is drawn after the 1-col left pad
            let drawn: String = row_text
                .chars()
                .skip(title_col)
                .take(title.chars().count())
                .collect();
            assert_eq!(
                drawn, title,
                "tab {tab:?}: title must render at hitbox.start+1 (col {title_col})"
            );
        }
    }

    #[test]
    fn narrow_terminal_drops_overflowing_tabs() {
        // At a width where not all tabs fit, the hit regions must not register a
        // tab ratatui didn't draw, every region is clamped to the area, and a
        // click well past the visible tabs resolves to nothing.
        let mut app = App::load().expect("App::load");
        let backend = TestBackend::new(20, 12);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw(frame, &mut app))
            .expect("draw must not panic");
        assert!(
            app.tab_hitboxes.len() < Tab::ALL.len(),
            "not all tabs fit at width 20"
        );
        for (_, end, _) in &app.tab_hitboxes {
            assert!(*end <= 20, "hit region clamped to the area width");
        }
        assert_eq!(app.tab_at(100, app.tab_row_y), None, "click far past the tabs");
    }

    #[test]
    fn help_overlay_does_not_panic_at_tiny_geometry() {
        // Locks in the saturating-sub underflow guards in draw_help_overlay.
        for (w, h) in [(4u16, 3u16), (2, 2), (1, 1)] {
            let mut app = App::load().expect("App::load");
            app.show_help = true;
            let backend = TestBackend::new(w, h);
            let mut terminal = Terminal::new(backend).expect("terminal");
            terminal
                .draw(|frame| draw(frame, &mut app))
                .expect("draw must not panic at tiny geometry");
        }
    }

    #[test]
    fn dashboard_footer_shows_serve_controls() {
        let text = render_with(|app| app.tab = Tab::Dashboard);
        assert!(text.contains("s start"), "serve start control in footer");
        assert!(text.contains("x stop"), "serve stop control in footer");
        assert!(text.contains("R restart"), "serve restart control in footer");
    }

    #[test]
    fn models_confirm_delete_prompt_renders() {
        let text = render_with(|app| {
            app.tab = Tab::Models;
            app.confirm_delete = Some("qwen3.5:9b".into());
        });
        assert!(text.contains("Confirm delete"), "confirm prompt title");
        assert!(text.contains("y to confirm"), "y/n guidance shown");
    }

    #[test]
    fn models_pull_progress_gauge_renders() {
        let text = render_with(|app| {
            app.tab = Tab::Models;
            let (_tx, rx) = std::sync::mpsc::channel();
            app.inject_pull("qwen3.5:9b".into(), rx);
            let job = app.pull.as_mut().unwrap();
            job.percent = Some(37.0);
            job.line = "[bar] 37.0% 8 MB/s".into();
        });
        assert!(text.contains("Pulling"), "gauge title shows the model");
        assert!(text.contains("37"), "percent in the gauge label");
    }

    #[test]
    fn logs_tab_missing_state_is_honest() {
        use crate::hipfire::log_tail::{LogSnapshot, LogStatus};
        let text = render_with(|app| {
            app.tab = Tab::Logs;
            app.logs = LogSnapshot {
                lines: vec![],
                status: LogStatus::Missing,
            };
        });
        assert!(text.contains("No serve.log yet"), "honest missing-state guidance");
    }

    #[test]
    fn system_tab_shows_request_inspector() {
        let text = render_with(|app| {
            app.tab = Tab::System;
            app.request_log = vec![crate::app::RequestRecord {
                model: "qwen3.5:9b".into(),
                tokens: 128,
                tps: 151.0,
                secs: 0.85,
            }];
        });
        assert!(text.contains("Recent requests"), "inspector section present");
        assert!(text.contains("128 tok"), "token count shown");
    }

    #[test]
    fn system_tab_renders_doctor_report() {
        use crate::hipfire::doctor::{DoctorCheck, DoctorReport};
        let text = render_with(|app| {
            app.tab = Tab::System;
            app.doctor = Some(DoctorReport {
                checks: vec![
                    DoctorCheck {
                        name: "amdgpu module".into(),
                        ok: true,
                        detail: "loaded".into(),
                    },
                    DoctorCheck {
                        name: "/dev/kfd".into(),
                        ok: false,
                        detail: "missing".into(),
                    },
                ],
                error: None,
            });
        });
        assert!(text.contains("Doctor (hipfire diag)"), "doctor section");
        assert!(text.contains("amdgpu module"), "check name shown");
        assert!(text.contains("missing"), "failed-check detail shown");
    }

    #[test]
    fn chat_renders_fenced_code_block() {
        let text = render_with(|app| {
            app.tab = Tab::Chat;
            app.chat.messages = vec![crate::hipfire::chat::ChatMessage {
                role: "assistant".into(),
                content: "here:\n```rust\nlet x = 1;\n```".into(),
            }];
        });
        assert!(text.contains("let x = 1;"), "code line is rendered");
        assert!(text.contains("rust"), "fence language label is shown");
    }

    #[test]
    fn logs_tab_renders_tail_lines() {
        use crate::hipfire::log_tail::{LogSnapshot, LogStatus};
        let text = render_with(|app| {
            app.tab = Tab::Logs;
            app.logs = LogSnapshot {
                lines: vec!["serve started on :11435".into(), "loaded qwen3.5:9b".into()],
                status: LogStatus::Ok,
            };
        });
        assert!(text.contains("loaded qwen3.5:9b"), "tail line is shown");
        assert!(text.contains("last 2 lines"), "title reflects the line count");
    }
}
