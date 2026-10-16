def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_score_version_and_factors(client):
    r = client.post("/predict", json={"features": {"n1": 0.5, "c1": "2"}})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["model_version"] == "baseline-lr-v1"
    assert len(body["top_factors"]) >= 1
    assert set(body["top_factors"][0]) == {"feature", "contribution"}


def test_predict_unseen_category_still_scores(client):
    r = client.post("/predict", json={"features": {"n1": 0.5, "c1": "Martian"}})
    assert r.status_code == 200


def test_predict_null_category_is_treated_as_missing(client):
    r = client.post("/predict", json={"features": {"n1": 0.5, "c1": None}})
    assert r.status_code == 200


def test_predict_integer_category_matches_string_category(client):
    a = client.post("/predict", json={"features": {"n1": 0.5, "c1": 2}}).json()["risk_score"]
    b = client.post("/predict", json={"features": {"n1": 0.5, "c1": "2"}}).json()["risk_score"]
    assert a == b


def test_predict_missing_feature_returns_422_naming_the_field(client):
    r = client.post("/predict", json={"features": {"n1": 0.5}})
    assert r.status_code == 422
    assert "c1" in r.text


def test_predict_non_numeric_value_returns_422(client):
    r = client.post("/predict", json={"features": {"n1": "abc", "c1": "2"}})
    assert r.status_code == 422


def test_predict_malformed_body_returns_422(client):
    r = client.post("/predict", json={"nope": 1})
    assert r.status_code == 422


def test_metrics_exposes_counter_and_histogram(client):
    client.post("/predict", json={"features": {"n1": 0.5, "c1": "2"}})
    text = client.get("/metrics").text
    assert "predict_requests_total" in text
    assert "predict_latency_seconds" in text


def test_service_without_model_starts_and_reports_503(client_no_model):
    assert client_no_model.get("/health").status_code == 503
    r = client_no_model.post("/predict", json={"features": {"n1": 0.5, "c1": "2"}})
    assert r.status_code == 503
