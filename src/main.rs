use axum::{
    routing::{get, post},
    Router,
    http::StatusCode,
    response::IntoResponse,
    response::Json,
};
use serde_json::json;
use serde::{Deserialize, Serialize};


#[derive(Deserialize)]
struct Payload {
    prompt: String,
}

#[derive(Serialize)]
struct Response {
    story: String,
}

#[tokio::main]
async fn main() {
    let server = "0.0.0.0:3000";
    // health check route
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/generate", post(generate_handler));

    // run app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind(server).await.unwrap();
    println!("Server running on http://{}", server);
    axum::serve(listener, app).await.unwrap();
}

async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "status": "ok" })))
}

async fn generate_handler(Json(payload): Json<Payload>) -> Json<Response> {
    Json(Response {
        story: payload.prompt,
    })
}