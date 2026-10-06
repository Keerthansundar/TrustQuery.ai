# Revenue Rules

- Revenue includes ONLY orders with status = 'completed'.
- Cancelled, returned and pending orders are excluded from revenue.
- All amounts are in INR (Indian Rupees), stored as whole rupees.
- `orders.revenue` is already the final amount for the whole order (quantity and discount applied). Do not multiply by quantity again.

## Dates
- order_date is TEXT in 'YYYY-MM-DD'.
- Filter a month with a half-open range: order_date >= '2026-09-01' AND order_date < '2026-10-01'.
- "Last month" means the latest complete month in the data (September 2026).
- "This year" means 2026. Relative dates are resolved against the latest data date, not the system clock.

## Comparing periods
- To explain a change between two months, compute both months in ONE query using conditional aggregation grouped by category (or product), and compute the difference and percent change in SQL.
- Percent change = (current - previous) * 100.0 / previous, rounded to 2 decimals.
