-- Write a SQL query to create a histogram of the number of comments per user in the month of January 2019. Assume bin buckets class intervals of one.


-- TABLE: users

-- columns	type
-- id	int
-- name	varchar
-- joined_at	datetime
-- city_id	int
-- device	int


-- TABLE: user_comments

-- columns	type
-- user_id	int
-- body	text
-- created_at	datetime



SELECT 
	users.id,
	COUNT (body),
FROM user_comments
WHERE created_at BETWEEN (jan 1, 2019) AND (jan 31, 2019) 
GROUP BY user_id;


-- ANSWER:


WITH hist AS (
	SELECT users.id, COUNT(user_comments.user_id) AS comment_count
	FROM users
	LEFT JOIN user_comments
	ON users.id = user_comments.user_id
	WHERE created_at BETWEEN '2019-01-01' AND '2019-01-31'
	GROUP BY 1)

SELECT comment_count, COUNT(*) AS frequency
FROM hist
GROUP BY 1





--------------------------------------------------------------------------------------------------------------------------------------------------------------------------





-- Given the revenue transactions table above that contains a user_id, created_at timestamp, and transaction revenue, write a query that finds the third purchase of every user

TABLE: transactions

columns	type
id	int
user_id	int
item	varchar
created_at	datetime
revenue	float




WITH CTE_VIEW AS (

SELECT 
	user_id,
	revenue,
	created_at,
	RANK() OVER(PARTITION BY user_id ORDER BY created_at ASC) Rank
FROM transactions
ORDER BY user_id ASC
)

SELECT 
	user_id,
	item
FROM CTE_VIEW
WHERE Rank = 3;



-- SELECT * 
-- FROM ( 
--   SELECT
--     user_id,
--     created_at,
--     revenue,
--     RANK() OVER (PARTITION BY user_id ORDER BY created_at ASC) AS rank_value 
--   FROM transactions
-- ) AS t
-- WHERE rank_value = 3;








--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Given three tables (user_dimension, account_dimension, and download_facts), find the average number of downloads for free vs. paying customers broken out by day.
-- Hint: The account_dimension table maps users to multiple accounts. They could be a paying customer or not in each account.



-- TABLE: user_dimension

-- columns	type
-- user_id	int
-- account_id	int
 

-- TABLE: account_dimension

-- columns	type
-- account_id	int
-- paying_customer	boolean
 

-- TABLE: download_facts

-- columns	type
-- user_id	int
-- date	date
-- downloads	int

SELECT 
	SUM(c.downloads) / COUNT(DISTINCT c.user_id) AS average_downloads
	c.date,
	b.paying_customer
FROM user_dimension a
INNER JOIN account_dimension b
ON a.account_id = b.account_id
LEFT JOIN download_facts c
ON c.user_id = a.user_id
GROUP BY 2, 3






















