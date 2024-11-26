FROM quay.io/astronomer/astro-runtime:12.4.0-base

# Switch to root user to install packages
USER root

# Install necessary packages
RUN apt-get update && apt-get install -y krb5-multidev libkrb5-dev build-essential

# Switch back to astro user
USER astro
