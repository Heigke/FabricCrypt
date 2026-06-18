#!/usr/bin/env python3
"""
Hypothalamus HTTP Server - Central Cluster Coordinator

Runs on the coordinator node and:
1. Receives state reports from all node agents
2. Provides routing decisions via API
3. Sends actuator commands to nodes
4. Exposes cluster state for dashboard

Usage:
    python hypothalamus_server.py --port 8765

API Endpoints:
    POST /node/<node_id>/state  - Receive node state report
    GET  /cluster/state         - Get full cluster state
    GET  /route                  - Get routing decision
    POST /node/<node_id>/command - Send command to node

Author: FEEL Research Team
Date: 2026-01-20
"""

import os
import sys
import json
import time
import argparse
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import urllib.request
import urllib.error

# Add project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from src.cluster.hypothalamus import (
    Hypothalamus, RoutingStrategy, NodeState, NodeStatus
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HypothalamusHandler(BaseHTTPRequestHandler):
    """HTTP handler for Hypothalamus API."""

    def log_message(self, format, *args):
        logger.debug(f"{self.address_string()} - {format % args}")

    def _send_json(self, data: dict, status: int = 200) -> None:
        """Send JSON response."""
        body = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        """Read JSON from request body."""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode('utf-8'))

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        hypo = self.server.hypothalamus

        if path == '/cluster/state':
            # Full cluster state
            state = hypo.get_cluster_state()
            self._send_json(state)

        elif path == '/route':
            # Get routing decision
            try:
                priority = float(query.get('priority', [0.5])[0])
                vendor = query.get('vendor', [None])[0]
                strategy = query.get('strategy', [None])[0]

                # Temporarily override strategy if specified
                original_strategy = hypo.strategy
                if strategy:
                    hypo.strategy = RoutingStrategy(strategy)

                decision = hypo.route_request(
                    task_priority=priority,
                    required_vendor=vendor,
                )

                hypo.strategy = original_strategy

                response = {
                    'target_node': decision.target_node,
                    'reason': decision.reason,
                    'fallback_nodes': decision.fallback_nodes,
                    'power_cap_adjustment': decision.power_cap_adjustment,
                    'profile_suggestion': decision.profile_suggestion,
                }
                self._send_json(response)

            except Exception as e:
                self._send_json({'error': str(e)}, 500)

        elif path == '/health':
            # Health check
            state = hypo.get_cluster_state()
            healthy = state['aggregates']['healthy_nodes']
            total = len(state['nodes'])

            self._send_json({
                'status': 'healthy' if healthy > 0 else 'degraded',
                'healthy_nodes': healthy,
                'total_nodes': total,
                'strategy': hypo.strategy.value,
            })

        elif path.startswith('/node/') and path.endswith('/state'):
            # Get specific node state
            node_id = path.split('/')[2]
            state = hypo.get_cluster_state()

            if node_id in state['nodes']:
                self._send_json(state['nodes'][node_id])
            else:
                self._send_json({'error': f'Unknown node: {node_id}'}, 404)

        else:
            self._send_json({'error': 'Not found'}, 404)

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        hypo = self.server.hypothalamus

        if path.startswith('/node/') and path.endswith('/state'):
            # Receive node state report
            node_id = path.split('/')[2]

            try:
                data = self._read_json()

                # Extract relevant fields
                state_update = {}
                for key in ['power_watts', 'temp_c', 'utilization', 'j_per_token',
                           'strain', 'urgency', 'debt', 'margin', 'stability',
                           'power_cap_watts', 'current_profile',
                           'requests_served', 'slo_violations',
                           'tbt_ms', 'throughput_tps']:
                    if key in data:
                        state_update[key] = data[key]

                hypo.update_node_state(node_id, state_update)

                logger.debug(f"Updated state for {node_id}: strain={state_update.get('strain', '?')}, "
                           f"j/tok={state_update.get('j_per_token', '?')}")

                self._send_json({'status': 'ok'})

            except Exception as e:
                logger.error(f"State update error: {e}")
                self._send_json({'error': str(e)}, 400)

        elif path.startswith('/node/') and path.endswith('/command'):
            # Send command to node
            node_id = path.split('/')[2]

            try:
                data = self._read_json()

                # Get node state to find its address
                cluster_state = hypo.get_cluster_state()
                if node_id not in cluster_state['nodes']:
                    self._send_json({'error': f'Unknown node: {node_id}'}, 404)
                    return

                node = cluster_state['nodes'][node_id]

                # Forward command to node agent
                # In production, would have node agent URLs in config
                # For now, just log
                logger.info(f"Command to {node_id}: {data}")

                self._send_json({'status': 'ok'})

            except Exception as e:
                self._send_json({'error': str(e)}, 400)

        elif path == '/strategy':
            # Change routing strategy
            try:
                data = self._read_json()
                strategy = data.get('strategy')
                if strategy:
                    hypo.strategy = RoutingStrategy(strategy)
                    logger.info(f"Strategy changed to: {strategy}")
                    self._send_json({'status': 'ok', 'strategy': strategy})
                else:
                    self._send_json({'error': 'Missing strategy'}, 400)
            except Exception as e:
                self._send_json({'error': str(e)}, 400)

        else:
            self._send_json({'error': 'Not found'}, 404)


class HypothalamusServer(HTTPServer):
    """HTTP server with Hypothalamus instance."""

    def __init__(self, address, handler, hypothalamus: Hypothalamus):
        super().__init__(address, handler)
        self.hypothalamus = hypothalamus


def create_cluster_config() -> list:
    """Create cluster configuration for ikaros/daedalus/minos."""
    return [
        {
            'node_id': 'ikaros',
            'hostname': '192.168.0.1',
            'port': 8766,
            'vendor': 'AMD',
        },
        {
            'node_id': 'daedalus',
            'hostname': '192.168.0.37',
            'port': 8766,
            'vendor': 'AMD',
        },
        {
            'node_id': 'minos',
            'hostname': '192.168.0.38',
            'port': 8766,
            'vendor': 'NVIDIA',
        },
    ]


def main():
    parser = argparse.ArgumentParser(description='FEEL Hypothalamus Server')
    parser.add_argument('--port', type=int, default=8765,
                       help='HTTP server port')
    parser.add_argument('--strategy', default='balanced',
                       choices=['coolest', 'most_efficient', 'least_loaded',
                               'round_robin', 'balanced'],
                       help='Initial routing strategy')
    parser.add_argument('--update-interval', type=int, default=1000,
                       help='Node update interval (ms)')

    args = parser.parse_args()

    # Create Hypothalamus with cluster config
    nodes = create_cluster_config()
    strategy = RoutingStrategy(args.strategy)

    hypothalamus = Hypothalamus(
        node_configs=nodes,
        strategy=strategy,
        update_interval_ms=args.update_interval,
    )

    # Start background updater
    hypothalamus.start()

    # Create and start HTTP server
    server = HypothalamusServer(
        ('0.0.0.0', args.port),
        HypothalamusHandler,
        hypothalamus
    )

    logger.info(f"Hypothalamus server starting on port {args.port}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Nodes: {[n['node_id'] for n in nodes]}")
    logger.info("Endpoints:")
    logger.info(f"  GET  http://localhost:{args.port}/cluster/state")
    logger.info(f"  GET  http://localhost:{args.port}/route")
    logger.info(f"  GET  http://localhost:{args.port}/health")
    logger.info(f"  POST http://localhost:{args.port}/node/<id>/state")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        hypothalamus.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
