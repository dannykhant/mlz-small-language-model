use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use ndarray::{Array2, s};
use ort::{session::Session, session::builder::GraphOptimizationLevel};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

pub struct TinyStoriesGenerator {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    input_name: String,
}

impl TinyStoriesGenerator {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Tokenizer load failed: {}", e))?;

        let input_name = session.inputs()[0].name().to_string();

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            input_name,
        })
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec();

        let mut rng = rand::thread_rng();

        for _ in 0..max_tokens {
            // Replicate the Python windowing: input_seq = np.array([input_ids[-127:]])
            let mut input_vec = vec![0i32; 127];
            let context = if tokens.len() > 127 {
                &tokens[tokens.len() - 127..]
            } else {
                &tokens
            };

            // Pre-padding logic (placing tokens at the end of the 127-window)
            let offset = 127 - context.len();
            for (i, &t) in context.iter().enumerate() {
                input_vec[offset + i] = t as i32;
            }

            // 1. Create the ndarray
            let input_array = Array2::from_shape_vec((1, 127), input_vec).expect("Shape mismatch");

            // 2. Wrap it in a Tensor
            let input_tensor = ort::value::Value::from_array(input_array)?;

            // 3. Run Inference and extract logits
            let exp_logits: Vec<f32> = {
                let mut session = self
                    .session
                    .lock()
                    .map_err(|e| anyhow::anyhow!("Mutex lock failed: {}", e))?;
                let outputs = session.run(ort::inputs![
                    self.input_name.as_str() => input_tensor
                ])?;
                let logits = outputs[0].try_extract_array::<half::f16>()?;
                let last_logits = logits.slice(s![0, -1, ..]);
                last_logits
                    .iter()
                    .map(|l| (l.to_f32() / temperature).exp())
                    .collect()
            };

            let sum_exp: f32 = exp_logits.iter().sum();

            let mut r = rng.r#gen::<f32>() * sum_exp;
            let mut next_token = 0;
            for (idx, &p) in exp_logits.iter().enumerate() {
                r -= p;
                if r <= 0.0 {
                    next_token = idx;
                    break;
                }
            }

            tokens.push(next_token as u32);
            if next_token == 3 {
                break;
            } // [EOS] token
        }

        self.tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!(e))
    }
}

// --- WEB SERVER INTEGRATION ---

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    max_tokens: Option<usize>,
}

#[derive(Serialize)]
struct GenerateResponse {
    story: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let generator = Arc::new(TinyStoriesGenerator::new(
        "models/tiny-stories-slm.onnx",
        "models/tiny_stories_tokenizer.json",
    )?);

    let app = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/generate", post(handle_generate))
        .with_state(generator);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("SLM serving at http://localhost:3000");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_generate(
    State(state): State<Arc<TinyStoriesGenerator>>,
    Json(payload): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    let story = state
        .generate(&payload.prompt, payload.max_tokens.unwrap_or(50), 0.7)
        .unwrap();

    Json(GenerateResponse { story })
}
