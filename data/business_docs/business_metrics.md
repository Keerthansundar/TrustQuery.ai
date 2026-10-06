# Business Metrics

- **Revenue**: SUM(orders.revenue) over completed orders.
- **Order count**: COUNT(*) of completed orders unless the user explicitly asks about other statuses.
- **Average order value (AOV)**: completed revenue divided by completed order count.
- **Cancellation rate**: cancelled orders * 100.0 / all orders in the period.
- **Return rate**: returned orders * 100.0 / all orders in the period.
- **Month-over-month growth %**: (this month revenue - previous month revenue) * 100.0 / previous month revenue.
- **Top customers**: rank customers by completed revenue, joining orders to customers on customer_id and
  grouping by customers.customer_id (customer names are not unique; never group by name alone).
- **New customers in a period**: customers whose signup_date falls in that period.
