-- KPI views on top of the star schema.
-- "delivered" below means order_status = 'delivered' and is_late is not null.

-- Monthly delivery KPIs
CREATE VIEW v_monthly_delivery_kpis AS
SELECT
    substr(o.purchase_ts, 1, 7)                        AS purchase_month,
    COUNT(*)                                           AS delivered_orders,
    ROUND(1.0 - AVG(o.is_late), 4)                     AS on_time_rate,
    ROUND(AVG(o.delivery_days), 2)                     AS avg_delivery_days,
    ROUND(AVG(o.promised_days), 2)                     AS avg_promised_days,
    ROUND(AVG(CASE WHEN o.is_late = 1 THEN o.delay_days END), 2) AS avg_days_late_when_late,
    ROUND(AVG(o.items_freight), 2)                     AS avg_freight
FROM fact_orders o
WHERE o.is_late IS NOT NULL
GROUP BY purchase_month
ORDER BY purchase_month;

-- Performance by customer state (where the package goes)
CREATE VIEW v_state_performance AS
SELECT
    c.state,
    c.state_name,
    COUNT(*)                            AS delivered_orders,
    ROUND(AVG(o.is_late), 4)            AS late_rate,
    ROUND(AVG(o.delivery_days), 2)      AS avg_delivery_days,
    ROUND(AVG(o.max_distance_km), 1)    AS avg_distance_km,
    ROUND(AVG(o.items_freight), 2)      AS avg_freight,
    ROUND(AVG(o.review_score), 2)       AS avg_review
FROM fact_orders o
JOIN dim_customer c ON c.customer_id = o.customer_id
WHERE o.is_late IS NOT NULL
GROUP BY c.state, c.state_name
ORDER BY late_rate DESC;

-- Seller state -> customer state lanes, only lanes with enough volume
CREATE VIEW v_lane_performance AS
SELECT
    s.state                             AS origin_state,
    c.state                             AS destination_state,
    COUNT(*)                            AS delivered_orders,
    ROUND(AVG(o.is_late), 4)            AS late_rate,
    ROUND(AVG(o.delivery_days), 2)      AS avg_delivery_days,
    ROUND(AVG(o.max_distance_km), 1)    AS avg_distance_km
FROM fact_orders o
JOIN dim_customer c ON c.customer_id = o.customer_id
JOIN dim_seller s   ON s.seller_id   = o.main_seller_id
WHERE o.is_late IS NOT NULL
GROUP BY s.state, c.state
HAVING COUNT(*) >= 30;

-- Product categories
CREATE VIEW v_category_performance AS
SELECT
    o.main_category_en                  AS category,
    COUNT(*)                            AS delivered_orders,
    ROUND(AVG(o.is_late), 4)            AS late_rate,
    ROUND(AVG(o.total_weight_g) / 1000.0, 2) AS avg_weight_kg,
    ROUND(AVG(o.items_freight), 2)      AS avg_freight,
    ROUND(AVG(o.review_score), 2)       AS avg_review
FROM fact_orders o
WHERE o.is_late IS NOT NULL
GROUP BY o.main_category_en
HAVING COUNT(*) >= 100
ORDER BY late_rate DESC;

-- Does being late hurt the review?
CREATE VIEW v_review_by_lateness AS
SELECT
    CASE WHEN o.is_late = 1 THEN 'late' ELSE 'on time' END AS delivery,
    COUNT(*)                                     AS orders_with_review,
    ROUND(AVG(o.review_score), 2)                AS avg_review,
    ROUND(AVG(CASE WHEN o.review_score <= 2 THEN 1.0 ELSE 0 END), 4) AS share_1_or_2_stars
FROM fact_orders o
WHERE o.is_late IS NOT NULL AND o.review_score IS NOT NULL
GROUP BY delivery;

-- Seller scorecard, sellers with at least 30 delivered orders
CREATE VIEW v_seller_scorecard AS
SELECT
    s.seller_id,
    s.state                             AS seller_state,
    COUNT(*)                            AS delivered_orders,
    ROUND(AVG(o.is_late), 4)            AS late_rate,
    ROUND(AVG(o.handling_days), 2)      AS avg_handling_days,
    ROUND(AVG(o.review_score), 2)       AS avg_review
FROM fact_orders o
JOIN dim_seller s ON s.seller_id = o.main_seller_id
WHERE o.is_late IS NOT NULL
GROUP BY s.seller_id, s.state
HAVING COUNT(*) >= 30;
