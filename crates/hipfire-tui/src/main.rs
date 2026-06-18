// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

mod app;
mod hipfire;
mod ui;

use std::{io, panic};

use anyhow::Result;
use app::App;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() -> Result<()> {
    // Non-interactive argument handling (headless-safe, no TTY required).
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--version" | "-V" => {
                println!("hipfire-tui {VERSION}");
                return Ok(());
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--check" => {
                return run_check();
            }
            other => {
                eprintln!("hipfire-tui: unknown argument '{other}'");
                print_help();
                std::process::exit(2);
            }
        }
    }

    let mut terminal = setup_terminal()?;
    let result = run(&mut terminal);
    restore_terminal(&mut terminal)?;
    result
}

fn print_help() {
    println!(
        "hipfire-tui {VERSION} - terminal UI for hipfire\n\
         \n\
         USAGE:\n    \
             hipfire-tui [FLAGS]\n\
         \n\
         FLAGS:\n    \
             -h, --help       Print this help and exit\n    \
             -V, --version    Print version and exit\n        \
                 --check      Load config/registry/models without entering the\n                         \
                              render loop, then exit 0 on success (headless smoke)\n\
         \n\
         With no flags, hipfire-tui launches the interactive ratatui UI (requires a TTY).\n\
         Tabs: Home, Dashboard, Chat, Models, Settings, System. Tab/BackTab to switch, q to quit."
    );
}

/// Construct the App state (config + registry + local models) WITHOUT entering
/// the ratatui render/event loop. Exits 0 on success, non-zero on init failure.
/// This is the headless smoke path for CI and TTY-less environments.
fn run_check() -> Result<()> {
    let app = App::load()?;
    println!("hipfire-tui --check OK");
    println!("  config: default_model = {:?}", app.config.default_model);
    println!("  registry: {} models, {} aliases", app.registry.models.len(), app.registry.aliases.len());
    println!("  local models: {}", app.registry.local_files.len());
    if let Some(warn) = &app.registry.warning {
        println!("  registry warning: {warn}");
    }
    Ok(())
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        hook(info);
    }));

    let backend = CrosstermBackend::new(stdout);
    Ok(Terminal::new(backend)?)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut app = App::load()?;

    loop {
        // Live-serve Dashboard: the fetch (HTTP + rocm-smi) runs on a dedicated
        // background thread. Here on the UI thread we ONLY mirror its latest
        // snapshot (a cheap lock+clone) and tell it whether the Dashboard tab
        // is focused — never any synchronous network / rocm-smi call, so a hung
        // probe cannot block render or input.
        app.sync_dashboard();
        app.expire_toast();

        terminal.draw(|frame| ui::draw(frame, &mut app))?;
        app.drain_chat_events();

        if event::poll(std::time::Duration::from_millis(80))? {
            match event::read()? {
                Event::Key(key) => {
                    if handle_key(&mut app, key) {
                        break;
                    }
                }
                Event::Resize(_, _) => {}
                _ => {}
            }
        }
    }

    Ok(())
}

fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        if app.chat.sending {
            app.chat.status =
                "stream is still running; wait for this spike build to finish it".into();
            return false;
        }
        return true;
    }

    // While a settings value is being edited, keystrokes (including q/e/a/r)
    // feed the edit buffer rather than triggering global shortcuts.
    let editing_setting = app.tab == app::Tab::Settings && app.settings_edit.is_some();
    if editing_setting && !matches!(key.code, KeyCode::Tab | KeyCode::BackTab) {
        app.handle_tab_key(key);
        return false;
    }

    // The chat input only captures global keys (q/r/Esc) while the Chat tab is
    // focused AND its input is focused. On every other tab the input is not on
    // screen, so q/r/Esc act as global shortcuts immediately from startup
    // (chat input defaults focused, but that must not gate other tabs).
    let chat_capturing = app.tab == app::Tab::Chat && app.chat.is_input_focused();

    match key.code {
        KeyCode::Char('q') if !chat_capturing => return true,
        KeyCode::Esc => {
            // While editing a settings value, Esc cancels the edit instead of
            // quitting / blurring chat.
            if app.tab == app::Tab::Settings && app.settings_edit.is_some() {
                app.handle_tab_key(key);
            } else if app.chat.sending {
                app.chat.status = "stream abort is not wired in prototype 1".into();
            } else if chat_capturing {
                // Only blur the chat input when the Chat tab is the one focused;
                // on other tabs Esc quits as before.
                app.chat.blur_input();
            } else {
                return true;
            }
        }
        KeyCode::Tab => app.next_tab(),
        KeyCode::BackTab => app.prev_tab(),
        KeyCode::Char('r') if !chat_capturing => app.reload(),
        KeyCode::Char('e') if app.tab == app::Tab::Settings => app.set_settings_easy(true),
        KeyCode::Char('a') if app.tab == app::Tab::Settings => app.set_settings_easy(false),
        _ => app.handle_tab_key(key),
    }

    false
}
