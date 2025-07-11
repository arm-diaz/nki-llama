#!/bin/bash

# Function to get metadata with retry
get_metadata() {
    local metadata_url="http://169.254.169.254/latest/meta-data"
    local max_attempts=5
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        # Try IMDSv2 first
        TOKEN=$(curl -s -f -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null)
        if [ $? -eq 0 ]; then
            RESULT=$(curl -s -f -H "X-aws-ec2-metadata-token: $TOKEN" $metadata_url/$1 2>/dev/null)
        else
            # Fallback to IMDSv1
            RESULT=$(curl -s -f $metadata_url/$1 2>/dev/null)
        fi

        if [ ! -z "$RESULT" ]; then
            echo "$RESULT"
            return 0
        fi

        echo "Attempt $attempt failed. Retrying..." >&2
        sleep 2
        ((attempt++))
    done

    echo "Failed to retrieve metadata after $max_attempts attempts" >&2
    return 1
}

# Get the instance type
INSTANCE_TYPE=$(get_metadata instance-type)

if [ $? -ne 0 ]; then
    echo "Failed to determine instance type"
    exit 1
fi

# Set the environment variable
export EC2_INSTANCE_TYPE=$INSTANCE_TYPE

# Print the instance type out
if [ "$EC2_INSTANCE_TYPE" == "trn1.2xlarge" ] || [ "$EC2_INSTANCE_TYPE" == "trn1.32xlarge" ]; then
    echo "======================================================"
    echo "✅ EC2_INSTANCE_TYPE: $EC2_INSTANCE_TYPE"
    echo "======================================================"
else
    echo "========================================================================================="
    echo "❌ This is not a trn1.2xlarge or trn1.32xlarge instance. It is a $INSTANCE_TYPE"
    echo "⚠️ Please use a valid instance type ⚠️"
    echo "========================================================================================="
fi
