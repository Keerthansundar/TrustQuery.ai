"""Generate a reproducible synthetic e-commerce database (amounts in INR).

Planted story for demos: August is a sale month (revenue spike) and September
sees a sharp drop in Phone demand, so "Why did revenue decrease in September?"
has a real, discoverable answer.

Run:  python scripts/seed_database.py
"""
import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_settings  # noqa: E402

SEED = 42
YEAR = 2026
BASE_MONTHLY_ORDERS = 620

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune", "Kolkata",
          "Ahmedabad", "Jaipur", "Kochi", "Mangaluru", "Lucknow"]
SEGMENTS = ["Consumer", "Small Business", "Enterprise"]
FIRST = ["Aarav", "Vivaan", "Aditya", "Ishaan", "Rohan", "Karthik", "Arjun", "Neha", "Priya", "Ananya",
         "Divya", "Meera", "Sneha", "Kavya", "Riya", "Rahul", "Sanjay", "Deepak", "Pooja", "Nikhil"]
LAST = ["Sharma", "Patel", "Nair", "Reddy", "Iyer", "Gupta", "Shetty", "Rao", "Kulkarni", "Singh",
        "Das", "Menon", "Bhat", "Joshi", "Khan"]

# (name, category, unit_price_inr)
PRODUCTS = [
    ("UltraBook Pro 14", "Laptop", 89999), ("WorkStation 16", "Laptop", 129999),
    ("StudyBook 15", "Laptop", 42999), ("GameForce 17", "Laptop", 109999),
    ("Nova X1", "Phone", 79999), ("Nova Lite", "Phone", 19999), ("Pixelon 9", "Phone", 64999),
    ("Orbit S", "Phone", 29999), ("Zenith Max", "Phone", 99999),
    ("TabPro 11", "Tablet", 54999), ("TabLite 10", "Tablet", 21999), ("TabMini 8", "Tablet", 15999),
    ("BassBuds Pro", "Audio", 7999), ("StudioPhones 200", "Audio", 12999), ("SoundBar 5.1", "Audio", 18999),
    ("Fast Charger 65W", "Accessories", 1999), ("Laptop Sleeve", "Accessories", 1499),
    ("USB-C Hub 7-in-1", "Accessories", 3499), ("Wireless Mouse", "Accessories", 1299),
    ("Mechanical Keyboard", "Accessories", 5999), ("Power Bank 20000", "Accessories", 2499),
]

CATEGORY_WEIGHT = {"Laptop": 0.15, "Phone": 0.30, "Tablet": 0.10, "Audio": 0.15, "Accessories": 0.30}
MONTH_VOLUME = {1: 0.90, 2: 0.85, 3: 1.00, 4: 0.95, 5: 1.00, 6: 1.00, 7: 1.05, 8: 1.15, 9: 1.00}
CATEGORY_MONTH = {("Phone", 9): 0.70, ("Laptop", 8): 1.10, ("Phone", 8): 1.15}
STATUSES = ["completed", "cancelled", "returned", "pending"]
STATUS_WEIGHTS = [0.82, 0.08, 0.06, 0.04]

SCHEMA = """
CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    city         TEXT NOT NULL,
    segment      TEXT NOT NULL CHECK (segment IN ('Consumer','Small Business','Enterprise')),
    signup_date  TEXT NOT NULL
);
CREATE TABLE products (
    product_id  INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    unit_price  INTEGER NOT NULL
);
CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    product_id   INTEGER NOT NULL REFERENCES products(product_id),
    order_date   TEXT NOT NULL,
    quantity     INTEGER NOT NULL,
    revenue      INTEGER NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('completed','cancelled','returned','pending'))
);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_product ON orders(product_id);
CREATE INDEX idx_orders_customer ON orders(customer_id);
"""


def _month_days(year: int, month: int) -> int:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return (nxt - date(year, month, 1)).days


def build_database(path: Path | str, seed: int = SEED) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rng = random.Random(seed)

    customers = []
    for cid in range(1, 401):
        signup = date(2024, 1, 1) + timedelta(days=rng.randint(0, 730))
        customers.append((cid, f"{rng.choice(FIRST)} {rng.choice(LAST)}", rng.choice(CITIES),
                          rng.choices(SEGMENTS, weights=[0.7, 0.2, 0.1])[0], signup.isoformat()))
    activity = [rng.paretovariate(1.5) for _ in customers]

    products = [(i, n, c, p) for i, (n, c, p) in enumerate(PRODUCTS, 1)]
    by_cat: dict[str, list] = {}
    for p in products:
        by_cat.setdefault(p[2], []).append(p)

    orders = []
    order_id = 1
    for month in range(1, 10):  # Jan..Sep 2026
        n_orders = round(BASE_MONTHLY_ORDERS * MONTH_VOLUME[month])
        cats = list(CATEGORY_WEIGHT)
        weights = [CATEGORY_WEIGHT[c] * CATEGORY_MONTH.get((c, month), 1.0) for c in cats]
        for _ in range(n_orders):
            cat = rng.choices(cats, weights=weights)[0]
            _, _, _, price = prod = rng.choice(by_cat[cat])
            cust = rng.choices(customers, weights=activity)[0]
            qty = rng.choices([1, 2, 3], weights=[0.75, 0.2, 0.05])[0] if cat == "Accessories" else \
                  rng.choices([1, 2], weights=[0.93, 0.07])[0]
            discount = rng.uniform(0.08, 0.12) if month == 8 else rng.uniform(0.0, 0.05)
            revenue = round(price * qty * (1 - discount))
            day = date(YEAR, month, rng.randint(1, _month_days(YEAR, month)))
            status = rng.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
            orders.append((order_id, cust[0], prod[0], day.isoformat(), qty, revenue, status))
            order_id += 1
    orders.sort(key=lambda o: o[3])
    orders = [(i, *o[1:]) for i, o in enumerate(orders, 1)]

    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", products)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", orders)
    conn.commit()
    conn.close()
    return path


def _summary(path: Path) -> None:
    conn = sqlite3.connect(path)
    print(f"Database created: {path}")
    for t in ("customers", "products", "orders"):
        print(f"  {t:<10} {conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>6} rows")
    print("\nCompleted-order revenue by month (INR):")
    for m, rev in conn.execute("SELECT substr(order_date,1,7), SUM(revenue) FROM orders "
                               "WHERE status='completed' GROUP BY 1 ORDER BY 1"):
        print(f"  {m}  {rev:>14,}")
    conn.close()


if __name__ == "__main__":
    out = build_database(get_settings().db_path)
    _summary(out)
