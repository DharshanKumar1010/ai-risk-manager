"""HMAC verification for inbound Razorpay webhooks.

``POST /webhooks/razorpay/transaction`` has no bearer token at all -- Razorpay does not carry
one, and there is no merchant JWT on this route by design (see ``app/api/webhooks.py``'s module
docstring for why that absence is what makes the route's response-disclosure exception safe).
The signature is the entire authentication surface, so it is checked over the **raw** request
body, before anything parses it.
"""

import hashlib
import hmac

from fastapi import HTTPException, status

#: Returned on any signature failure -- missing header, wrong secret, or a tampered body.
#: Deliberately uninformative, mirroring decode_access_token's "a 401 says no, not which no"
#: (security-checklist item 2.5): telling a caller *why* verification failed would tell an
#: attacker which part of a forged request to fix next.
INVALID_SIGNATURE_DETAIL = "Invalid webhook signature"


def verify_razorpay_signature(body: bytes, signature: str | None, secret: str) -> None:
    """Verify ``X-Razorpay-Signature`` over the raw, unparsed request body.

    Args:
        body: The exact bytes Razorpay signed. Must be read before FastAPI/Pydantic parses
            the request, since parsing and re-serialising would not reproduce the same bytes.
        signature: The header value, or ``None`` if absent.
        secret: This deployment's shared webhook secret (``settings.razorpay_webhook_secret``).

    Raises:
        HTTPException: 401, with :data:`INVALID_SIGNATURE_DETAIL`, on a missing header or a
            mismatch.
    """
    if not signature:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_SIGNATURE_DETAIL)

    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    # hmac.compare_digest, never ==: a short-circuiting comparison would leak the signature's
    # correct prefix length through response timing.
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_SIGNATURE_DETAIL)
