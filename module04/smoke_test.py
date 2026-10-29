"""Smoke test for the local inference container; standard library only."""
import argparse
import json
import math
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def get_json(url: str):
    with urlopen(url, timeout=15) as response:
        return response.status, json.load(response)


def post_json(url: str, data: dict):
    request = Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urlopen(request, timeout=15) as response:
        return response.status, json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    parser.add_argument('--expected', type=float, default=18.61)
    args = parser.parse_args()

    base = args.url.rstrip('/')
    code, health = get_json(base + '/health')
    assert code == 200 and health['status'] == 'ok', (code, health)
    print('health:', health)

    trip = {'PULocationID': 161, 'DOLocationID': 141, 'trip_distance': 3.5}
    code, result = post_json(base + '/predict', trip)
    value = result['duration_minutes']
    assert code == 200 and math.isfinite(value), (code, result)
    assert abs(value - args.expected) <= 0.05, (
        f'Expected approximately {args.expected}, got {value}. '
        'Confirm model/vectorizer pair and versions.'
    )
    print('prediction:', result)

    try:
        post_json(base + '/predict', {**trip, 'trip_distance': -1})
    except HTTPError as exc:
        assert exc.code == 422, f'Expected 422 for invalid distance, got {exc.code}'
        print('invalid request: HTTP 422 (expected)')
    else:
        raise AssertionError('Invalid request was not rejected')
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
