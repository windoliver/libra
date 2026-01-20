//! Order Book Widget using Ratatui.
//!
//! High-performance order book rendering for latency-critical display.
//! Uses Ratatui's zero-cost abstractions for sub-millisecond rendering.
//!
//! Issue #101: Evaluate Ratatui for TUI widgets.

use pyo3::prelude::*;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Widget},
};

/// Order book level with price and quantity.
#[derive(Clone, Debug)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
    pub total: f64, // Cumulative quantity
}

/// Order book widget for rendering bid/ask levels.
#[pyclass]
pub struct OrderBookWidget {
    bids: Vec<OrderBookLevel>,
    asks: Vec<OrderBookLevel>,
    max_levels: usize,
    spread: f64,
    mid_price: f64,
}

#[pymethods]
impl OrderBookWidget {
    /// Create a new order book widget.
    #[new]
    #[pyo3(signature = (max_levels=20))]
    pub fn new(max_levels: usize) -> Self {
        Self {
            bids: Vec::with_capacity(max_levels),
            asks: Vec::with_capacity(max_levels),
            max_levels,
            spread: 0.0,
            mid_price: 0.0,
        }
    }

    /// Update the order book with new levels.
    ///
    /// Args:
    ///     bid_prices: List of bid prices (highest first)
    ///     bid_quantities: List of bid quantities
    ///     ask_prices: List of ask prices (lowest first)
    ///     ask_quantities: List of ask quantities
    pub fn update(
        &mut self,
        bid_prices: Vec<f64>,
        bid_quantities: Vec<f64>,
        ask_prices: Vec<f64>,
        ask_quantities: Vec<f64>,
    ) {
        // Update bids
        self.bids.clear();
        let mut cumulative = 0.0;
        for (price, qty) in bid_prices.iter().zip(bid_quantities.iter()).take(self.max_levels) {
            cumulative += qty;
            self.bids.push(OrderBookLevel {
                price: *price,
                quantity: *qty,
                total: cumulative,
            });
        }

        // Update asks
        self.asks.clear();
        cumulative = 0.0;
        for (price, qty) in ask_prices.iter().zip(ask_quantities.iter()).take(self.max_levels) {
            cumulative += qty;
            self.asks.push(OrderBookLevel {
                price: *price,
                quantity: *qty,
                total: cumulative,
            });
        }

        // Calculate spread and mid price
        if let (Some(best_bid), Some(best_ask)) = (self.bids.first(), self.asks.first()) {
            self.spread = best_ask.price - best_bid.price;
            self.mid_price = (best_bid.price + best_ask.price) / 2.0;
        }
    }

    /// Render the order book to ANSI string for terminal display.
    ///
    /// Returns:
    ///     ANSI-formatted string for terminal rendering
    pub fn render_ansi(&self, width: u16, height: u16) -> String {
        // Create a buffer for rendering
        let area = Rect::new(0, 0, width, height);
        let mut buffer = Buffer::empty(area);

        // Render widget to buffer
        self.render_to_buffer(area, &mut buffer);

        // Convert buffer to ANSI string
        buffer_to_ansi(&buffer)
    }

    /// Get the current spread.
    pub fn get_spread(&self) -> f64 {
        self.spread
    }

    /// Get the current mid price.
    pub fn get_mid_price(&self) -> f64 {
        self.mid_price
    }

    /// Get number of bid levels.
    pub fn bid_count(&self) -> usize {
        self.bids.len()
    }

    /// Get number of ask levels.
    pub fn ask_count(&self) -> usize {
        self.asks.len()
    }
}

impl OrderBookWidget {
    /// Render to a Ratatui buffer.
    fn render_to_buffer(&self, area: Rect, buf: &mut Buffer) {
        // Calculate layout
        let inner_height = area.height.saturating_sub(2); // Account for borders
        let half_height = inner_height / 2;

        // Find max quantity for bar scaling
        let max_qty = self
            .bids
            .iter()
            .chain(self.asks.iter())
            .map(|l| l.quantity)
            .fold(0.0_f64, f64::max);

        let bar_width = area.width.saturating_sub(30) as f64; // Space for price/qty columns

        // Draw border
        let block = Block::default()
            .title(format!(
                " Order Book | Spread: {:.2} | Mid: {:.2} ",
                self.spread, self.mid_price
            ))
            .borders(Borders::ALL);
        let inner = block.inner(area);
        block.render(area, buf);

        // Draw asks (top half, reversed so lowest ask is at bottom)
        for (i, level) in self.asks.iter().take(half_height as usize).enumerate() {
            let y = inner.y + (half_height.saturating_sub(1) - i as u16);
            if y >= inner.y && y < inner.y + inner.height {
                self.render_level(buf, inner.x, y, inner.width, level, true, max_qty, bar_width);
            }
        }

        // Draw spread line
        let spread_y = inner.y + half_height;
        if spread_y < inner.y + inner.height {
            let spread_text = format!("─── Spread: {:.4} ───", self.spread);
            let x = inner.x + (inner.width.saturating_sub(spread_text.len() as u16)) / 2;
            buf.set_string(x, spread_y, &spread_text, Style::default().fg(Color::Yellow));
        }

        // Draw bids (bottom half)
        for (i, level) in self.bids.iter().take(half_height as usize).enumerate() {
            let y = inner.y + half_height + 1 + i as u16;
            if y < inner.y + inner.height {
                self.render_level(buf, inner.x, y, inner.width, level, false, max_qty, bar_width);
            }
        }
    }

    /// Render a single order book level.
    fn render_level(
        &self,
        buf: &mut Buffer,
        x: u16,
        y: u16,
        _width: u16,
        level: &OrderBookLevel,
        is_ask: bool,
        max_qty: f64,
        bar_width: f64,
    ) {
        let color = if is_ask { Color::Red } else { Color::Green };

        // Format price and quantity
        let price_str = format!("{:>12.4}", level.price);
        let qty_str = format!("{:>10.4}", level.quantity);
        let total_str = format!("{:>10.4}", level.total);

        // Calculate bar length
        let bar_len = if max_qty > 0.0 {
            ((level.quantity / max_qty) * bar_width) as usize
        } else {
            0
        };
        let bar = "█".repeat(bar_len);

        // Render components
        buf.set_string(x, y, &price_str, Style::default().fg(color));
        buf.set_string(x + 13, y, &qty_str, Style::default().fg(Color::White));
        buf.set_string(x + 24, y, &total_str, Style::default().fg(Color::DarkGray));
        buf.set_string(x + 35, y, &bar, Style::default().fg(color));
    }
}

/// Convert a Ratatui buffer to ANSI escape sequence string.
fn buffer_to_ansi(buffer: &Buffer) -> String {
    let mut output = String::with_capacity(buffer.area.width as usize * buffer.area.height as usize * 20);

    for y in buffer.area.y..buffer.area.y + buffer.area.height {
        if y > buffer.area.y {
            output.push('\n');
        }

        let mut last_style = Style::default();

        for x in buffer.area.x..buffer.area.x + buffer.area.width {
            let cell = &buffer[(x, y)];

            // Apply style changes
            if cell.style() != last_style {
                // Reset and apply new style
                output.push_str("\x1b[0m");

                if let Some(fg) = cell.style().fg {
                    output.push_str(&color_to_ansi_fg(fg));
                }
                if let Some(bg) = cell.style().bg {
                    output.push_str(&color_to_ansi_bg(bg));
                }

                last_style = cell.style();
            }

            output.push_str(cell.symbol());
        }

        // Reset at end of line
        output.push_str("\x1b[0m");
    }

    output
}

/// Convert Ratatui color to ANSI foreground escape sequence.
fn color_to_ansi_fg(color: Color) -> String {
    match color {
        Color::Black => "\x1b[30m".to_string(),
        Color::Red => "\x1b[31m".to_string(),
        Color::Green => "\x1b[32m".to_string(),
        Color::Yellow => "\x1b[33m".to_string(),
        Color::Blue => "\x1b[34m".to_string(),
        Color::Magenta => "\x1b[35m".to_string(),
        Color::Cyan => "\x1b[36m".to_string(),
        Color::Gray | Color::White => "\x1b[37m".to_string(),
        Color::DarkGray => "\x1b[90m".to_string(),
        Color::LightRed => "\x1b[91m".to_string(),
        Color::LightGreen => "\x1b[92m".to_string(),
        Color::LightYellow => "\x1b[93m".to_string(),
        Color::LightBlue => "\x1b[94m".to_string(),
        Color::LightMagenta => "\x1b[95m".to_string(),
        Color::LightCyan => "\x1b[96m".to_string(),
        Color::Rgb(r, g, b) => format!("\x1b[38;2;{};{};{}m", r, g, b),
        Color::Indexed(i) => format!("\x1b[38;5;{}m", i),
        _ => String::new(),
    }
}

/// Convert Ratatui color to ANSI background escape sequence.
fn color_to_ansi_bg(color: Color) -> String {
    match color {
        Color::Black => "\x1b[40m".to_string(),
        Color::Red => "\x1b[41m".to_string(),
        Color::Green => "\x1b[42m".to_string(),
        Color::Yellow => "\x1b[43m".to_string(),
        Color::Blue => "\x1b[44m".to_string(),
        Color::Magenta => "\x1b[45m".to_string(),
        Color::Cyan => "\x1b[46m".to_string(),
        Color::Gray | Color::White => "\x1b[47m".to_string(),
        Color::DarkGray => "\x1b[100m".to_string(),
        Color::Rgb(r, g, b) => format!("\x1b[48;2;{};{};{}m", r, g, b),
        Color::Indexed(i) => format!("\x1b[48;5;{}m", i),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_update() {
        let mut widget = OrderBookWidget::new(10);

        widget.update(
            vec![100.0, 99.9, 99.8],
            vec![1.0, 2.0, 3.0],
            vec![100.1, 100.2, 100.3],
            vec![1.5, 2.5, 3.5],
        );

        assert_eq!(widget.bid_count(), 3);
        assert_eq!(widget.ask_count(), 3);
        assert!((widget.get_spread() - 0.1).abs() < 0.0001);
        assert!((widget.get_mid_price() - 100.05).abs() < 0.0001);
    }

    #[test]
    fn test_render_ansi() {
        let mut widget = OrderBookWidget::new(5);

        widget.update(
            vec![100.0, 99.9],
            vec![1.0, 2.0],
            vec![100.1, 100.2],
            vec![1.5, 2.5],
        );

        let output = widget.render_ansi(80, 20);
        assert!(!output.is_empty());
        assert!(output.contains("Spread"));
    }
}
