
#!/usr/bin/env bash

# Initialize the database
airflow db init

# Run the command
exec "$@"