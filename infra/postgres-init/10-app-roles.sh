#!/bin/bash
# Create the least-privilege application roles with passwords from the environment.
#
# Why this exists, and why it is a shell script rather than part of the Alembic migration:
# a migration is tracked source, and a role password in tracked source is the exact failure
# the secret scan exists to catch. So revision 0002 grants LOGIN and defines the policies,
# and the credential is attached here, from the environment, at first boot.
#
# Revision 0001 creates these roles with `IF NOT EXISTS`, so this script and the migration
# compose in either order: whichever runs first creates the role, the other leaves it alone.
#
# The official postgres image runs everything in /docker-entrypoint-initdb.d once, on an
# empty data directory. An existing volume is left untouched -- change a password on a
# running stack with ALTER ROLE, not by editing this file.
set -euo pipefail

if [[ -z "${RISKIQ_APP_PASSWORD:-}" || -z "${RISKIQ_PIPELINE_PASSWORD:-}" ]]; then
    echo "10-app-roles.sh: RISKIQ_APP_PASSWORD and RISKIQ_PIPELINE_PASSWORD must be set" >&2
    exit 1
fi

# Passwords are passed as psql variables and quoted by the server, never interpolated into
# the SQL text by the shell. :'name' is psql's quote-as-literal form.
psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     --set=app_password="$RISKIQ_APP_PASSWORD" \
     --set=pipeline_password="$RISKIQ_PIPELINE_PASSWORD" \
     --no-psqlrc --quiet --set=ON_ERROR_STOP=1 <<'SQL'
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'riskiq_app') THEN
        CREATE ROLE riskiq_app NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'riskiq_pipeline') THEN
        CREATE ROLE riskiq_pipeline NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'riskiq_analyst') THEN
        CREATE ROLE riskiq_analyst NOLOGIN;
    END IF;
END
$$;

ALTER ROLE riskiq_app      LOGIN PASSWORD :'app_password';
ALTER ROLE riskiq_pipeline LOGIN PASSWORD :'pipeline_password';
SQL

echo "10-app-roles.sh: riskiq_app and riskiq_pipeline can now log in"
