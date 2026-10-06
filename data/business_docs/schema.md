# Database Schema (SQLite)

## customers

One row per customer.

- customer_id (INTEGER, PK)
- name (TEXT) - NOT unique: several customers share the same name. Identify customers by customer_id.
- city (TEXT) - e.g. Mumbai, Bengaluru, Mangaluru
- segment (TEXT) - one of: Consumer, Small Business, Enterprise
- signup_date (TEXT, 'YYYY-MM-DD')

## products

One row per product.

- product_id (INTEGER, PK)
- name (TEXT) - product name, e.g. "Nova X1"
- category (TEXT) - Laptop, Phone, Tablet, Audio, Accessories
- unit_price (INTEGER) - list price in INR (whole rupees)

## orders

One row per order. Each order contains exactly one product.

- order_id (INTEGER, PK)
- customer_id (INTEGER, FK -> customers.customer_id)
- product_id (INTEGER, FK -> products.product_id)
- order_date (TEXT, 'YYYY-MM-DD')
- quantity (INTEGER)
- revenue (INTEGER) - order value in INR (whole rupees), after discount, quantity included
- status (TEXT) - completed, cancelled, returned, pending

## Join paths

- orders.product_id = products.product_id
- orders.customer_id = customers.customer_id

## Data window

Orders cover 2026-01-01 to 2026-09-30.
