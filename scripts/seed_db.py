import sys
from pathlib import Path as _BootstrapPath

ROOT_BOOTSTRAP = _BootstrapPath(__file__).resolve().parents[1]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))

import json
from pathlib import Path

from sqlalchemy.exc import OperationalError, ProgrammingError

from backend.app.db import SessionLocal
from backend.app.models import Product, ProductImage

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "data/demo/catalog.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    db = SessionLocal()
    try:
        for row in rows:
            if db.get(Product, row["id"]):
                continue
            db.add(
                Product(
                    id=row["id"],
                    title=row["title"],
                    description=row["description"],
                    category=row["category"],
                    brand=row["brand"],
                    price=row["price"],
                    colour=row["colour"],
                    attributes=row["attributes"],
                    tags=row["tags"],
                    image_path=row["image_path"],
                )
            )
            db.add(ProductImage(product_id=row["id"], path=row["image_path"]))
        db.commit()
    except (OperationalError, ProgrammingError) as exc:
        db.rollback()
        raise RuntimeError("Database schema is missing. Run 'alembic -c backend/alembic.ini upgrade head' before seeding.") from exc
    finally:
        db.close()
    print(f"seeded/verified {len(rows)} demo products")


if __name__ == "__main__":
    main()
