#!/bin/sh
# Read-only role for the controlled analytics-SQL mode. Table-level SELECT grants on the
# allowlisted tables are applied by scripts/bootstrap_demo.py after the schema exists.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
  CREATE ROLE analytics_ro LOGIN PASSWORD '${ANALYTICS_RO_PASSWORD:-analytics_ro}'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
  ALTER ROLE analytics_ro SET default_transaction_read_only = on;
  ALTER ROLE analytics_ro SET statement_timeout = '3s';
  GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO analytics_ro;
  GRANT USAGE ON SCHEMA public TO analytics_ro;
SQL
