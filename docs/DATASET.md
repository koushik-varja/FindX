# Demo dataset

The repository includes 82 generated commerce records and matching locally generated demo images. Categories cover shoes, apparel, electronics, bags, watches and home products.

The records are synthetic demonstration data intended for search-system engineering. Brand names are used as catalog text examples; the generated product titles/images do not assert affiliation with those brands.

Each JSONL record contains `id`, `title`, `description`, `category`, `brand`, `price`, `colour`, `attributes`, `tags`, `image_path` and a small language-search field where applicable.

The generated images are simple product-style illustrations. They make image-similarity behaviour reproducible without downloading photographs or relying on a remote CDN.
