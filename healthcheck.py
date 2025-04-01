#!/usr/bin/env python3
"""
Health check script for Docker
"""

import sys
import requests
import argparse

def check_service(url, service_name):
    """Check if a service is responding"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"{service_name} is healthy")
            return True
        else:
            print(f"{service_name} returned status code {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Error connecting to {service_name}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check service health')
    parser.add_argument('--service', choices=['agent', 'dashboard'], required=True,
                        help='Service to check (agent or dashboard)')
    
    args = parser.parse_args()
    
    if args.service == 'agent':
        success = check_service('http://localhost:8000/status', 'Trade Agent API')
    else:
        success = check_service('http://localhost:8080/', 'Dashboard')
    
    sys.exit(0 if success else 1)