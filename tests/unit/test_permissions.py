import pytest

from packages.core.security import (
    ACCESS_LEVELS,
    Permission,
    Principal,
    Role,
    create_token,
    decode_token,
    hash_password,
    verify_password,
)


@pytest.mark.parametrize(
    "role,allowed,denied",
    [
        (
            Role.EMPLOYEE,
            [Permission.DOCS_READ, Permission.PRODUCTS_SEARCH],
            [
                Permission.STOCK_READ,
                Permission.PRICING_READ,
                Permission.CONTENT_GENERATE,
                Permission.UPDATES_APPROVE,
            ],
        ),
        (
            Role.SALES,
            [Permission.STOCK_READ, Permission.PRICING_READ],
            [Permission.CONTENT_GENERATE, Permission.UPDATES_REQUEST, Permission.ANALYTICS_SQL],
        ),
        (
            Role.CATALOG_MANAGER,
            [Permission.CONTENT_GENERATE, Permission.UPDATES_REQUEST, Permission.BATCH_READ],
            [Permission.UPDATES_APPROVE, Permission.BATCH_RUN, Permission.ANALYTICS_SQL],
        ),
        (Role.ADMIN, list(Permission), []),
    ],
)
def test_role_permission_matrix(role, allowed, denied):
    p = Principal("u", role)
    assert all(p.can(x) for x in allowed)
    assert not any(p.can(x) for x in denied)


def test_access_levels_are_nested_least_privilege():
    assert set(ACCESS_LEVELS["management"]) == {"admin"}
    assert "employee" not in ACCESS_LEVELS["commercial"]
    assert set(ACCESS_LEVELS["all_staff"]) == {r.value for r in Role}


def test_token_roundtrip_and_password_hashing():
    p = Principal("samuel.sales", Role.SALES, "Samuel")
    assert decode_token(create_token(p)) == p
    h = hash_password("demo")
    assert verify_password("demo", h) and not verify_password("wrong", h)
    assert hash_password("demo") != h  # salted


def test_tampered_token_is_rejected():
    import jwt

    token = create_token(Principal("eva.employee", Role.EMPLOYEE))
    forged = jwt.encode(
        {**jwt.decode(token, options={"verify_signature": False}), "role": "admin"},
        "wrong-secret-xyz-123",
        algorithm="HS256",
    )
    with pytest.raises(jwt.InvalidSignatureError):
        decode_token(forged)
