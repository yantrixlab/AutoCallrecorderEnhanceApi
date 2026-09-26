"""Server-side verification of Play Billing purchases via the Android
Publisher API. This exists because a tampered APK could trivially fake a
local "isPremium = true" flag - the actual purchase token the app receives
from Play Billing has to be checked against Google's own records before
anything premium is unlocked for real."""

import json
import logging
import os

from google.oauth2 import service_account
from googleapiclient.discovery import build

logger = logging.getLogger("enhance_api")

PACKAGE_NAME = "com.yantrixlab.autocallrecorder"
SCOPES = ["https://www.googleapis.com/auth/androidpublisher"]

_service = None


def _get_service():
    """Lazily built and cached - avoids re-parsing the service account key
    and re-authenticating on every single verification request."""
    global _service
    if _service is not None:
        return _service

    raw_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not set")

    info = json.loads(raw_json)
    credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    _service = build("androidpublisher", "v3", credentials=credentials, cache_discovery=False)
    return _service


def verify_subscription(product_id: str, purchase_token: str) -> dict:
    """Uses the v2 subscriptions API (Google's current recommendation over
    the deprecated v1 purchases.subscriptions.get). A subscription is only
    valid if its state is explicitly ACTIVE or IN_GRACE_PERIOD (still
    entitled while Google retries a failed renewal payment) - anything else
    (canceled, expired, on hold, paused) means access should not be granted."""
    service = _get_service()
    result = service.purchases().subscriptionsv2().get(
        packageName=PACKAGE_NAME, token=purchase_token
    ).execute()

    state = result.get("subscriptionState", "")
    valid = state in ("SUBSCRIPTION_STATE_ACTIVE", "SUBSCRIPTION_STATE_IN_GRACE_PERIOD")

    expiry_time_millis = None
    for line_item in result.get("lineItems", []):
        if line_item.get("productId") == product_id and "expiryTime" in line_item:
            expiry_time_millis = line_item["expiryTime"]

    return {
        "valid": valid,
        "subscription_state": state,
        "expiry_time": expiry_time_millis,
    }


def verify_one_time_product(product_id: str, purchase_token: str) -> dict:
    """purchaseState: 0 = Purchased, 1 = Canceled, 2 = Pending - only 0 grants
    access. A pending purchase (e.g. awaiting a slow payment method) must not
    unlock anything until Google confirms it actually completed."""
    service = _get_service()
    result = service.purchases().products().get(
        packageName=PACKAGE_NAME, productId=product_id, token=purchase_token
    ).execute()

    purchase_state = result.get("purchaseState")
    valid = purchase_state == 0

    return {
        "valid": valid,
        "purchase_state": purchase_state,
        "purchase_time_millis": result.get("purchaseTimeMillis"),
    }
