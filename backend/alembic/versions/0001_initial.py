"""initial schema
Revision ID: 0001
Revises:
"""
from alembic import op
import sqlalchemy as sa
revision='0001'; down_revision=None; branch_labels=None; depends_on=None
def upgrade():
    op.create_table('products',sa.Column('id',sa.String(32),primary_key=True),sa.Column('title',sa.String(240),nullable=False),sa.Column('description',sa.Text(),nullable=False),sa.Column('category',sa.String(80)),sa.Column('brand',sa.String(80)),sa.Column('price',sa.Integer()),sa.Column('colour',sa.String(40)),sa.Column('attributes',sa.JSON(),nullable=False),sa.Column('tags',sa.JSON(),nullable=False),sa.Column('image_path',sa.String(300),nullable=False))
    for name,col in [('ix_products_title','title'),('ix_products_category','category'),('ix_products_brand','brand'),('ix_products_price','price'),('ix_products_colour','colour')]: op.create_index(name,'products',[col])
    op.create_table('product_images',sa.Column('id',sa.Integer(),primary_key=True,autoincrement=True),sa.Column('product_id',sa.String(32),sa.ForeignKey('products.id',ondelete='CASCADE'),nullable=False),sa.Column('path',sa.String(300),nullable=False)); op.create_index('ix_product_images_product_id','product_images',['product_id'])
    op.create_table('search_queries',sa.Column('id',sa.String(64),primary_key=True),sa.Column('original_query',sa.Text(),nullable=False),sa.Column('normalized_query',sa.Text(),nullable=False),sa.Column('parsed_attributes',sa.JSON(),nullable=False),sa.Column('mode',sa.String(32),nullable=False),sa.Column('latency_ms',sa.Float(),nullable=False),sa.Column('created_at',sa.DateTime(timezone=True),server_default=sa.func.now())); op.create_index('ix_search_queries_created_at','search_queries',['created_at'])
    op.create_table('search_events',sa.Column('id',sa.Integer(),primary_key=True,autoincrement=True),sa.Column('query_id',sa.String(64),sa.ForeignKey('search_queries.id',ondelete='CASCADE'),nullable=False),sa.Column('product_id',sa.String(32),nullable=False),sa.Column('rank',sa.Integer(),nullable=False),sa.Column('score',sa.Float(),nullable=False)); op.create_index('ix_search_events_query_id','search_events',['query_id'])
    op.create_table('index_versions',sa.Column('id',sa.Integer(),primary_key=True,autoincrement=True),sa.Column('version',sa.String(120),nullable=False),sa.Column('product_count',sa.Integer(),nullable=False),sa.Column('dense_model',sa.String(200),nullable=False),sa.Column('visual_model',sa.String(200),nullable=False),sa.Column('build_seconds',sa.Float(),nullable=False),sa.Column('created_at',sa.DateTime(timezone=True),server_default=sa.func.now())); op.create_index('ix_index_versions_version','index_versions',['version'])
def downgrade():
    op.drop_table('index_versions'); op.drop_table('search_events'); op.drop_table('search_queries'); op.drop_table('product_images'); op.drop_table('products')
