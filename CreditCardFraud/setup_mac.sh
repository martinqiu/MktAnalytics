#!/bin/bash
# Move to the directory where the script is located
cd "$(dirname "$0")"

echo "================================================="
echo "   Credit Card Fraud Dashboard Setup (Mac/Linux)"
echo "================================================="
echo ""

# Check if pip3 is installed
if ! command -v pip3 &> /dev/null
then
    echo "ERROR: pip3 could not be found. Please install Python 3."
    exit
fi

# Install requirements
echo "Installing required libraries..."
pip3 install -r requirements.txt

echo ""
echo "================================================="
echo "   Setup Complete! Launching Dashboard..."
echo "================================================="
echo ""

# Run the dashboard
streamlit run fraud_dashboard.py
