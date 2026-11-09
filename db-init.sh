#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER bookaudio_app WITH ENCRYPTED PASSWORD '$BOOKAUDIO_APP_PASSWORD';
    CREATE DATABASE bookaudio;
    GRANT CONNECT ON DATABASE bookaudio TO bookaudio_app;
    GRANT ALL PRIVILEGES ON DATABASE bookaudio TO bookaudio_app;
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "bookaudio" <<-EOSQL
    GRANT ALL ON SCHEMA public TO bookaudio_app;
EOSQL
