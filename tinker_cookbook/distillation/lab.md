ultrathink

# LLM 모델 SFT fine-tuning
- /workspace/tinker-cookbook-arisohn/tinker_cookbook/distillation.venv 환경을 사용하세요.
- 학습데이터는 /workspace/datasets--open-thoughts--OpenThoughts3-1.2M-shuffle-1k 를 사용하세요.
- 학습모델은 /workspace/models--Qwen--Qwen3-8B-Base 를 사용하세요.
- huggingface trl 라이브러리를 사용하세요.
- loss rate는 5e-5 로 scheduler는 cosine을 사용하세요.
- 3 epoch 학습하세요.
- 학습코드 학습결과 및 학습로그는 /workspace/trl 에 저장해주세요.
- flash attention을 사용하세요
- full fine tunning을 하세요.
- max length는 8192 입니다.
- precision은 bf16을 사용해주세요.
- 유저의 결정이 필요한 사항들을 유저에게 문의하세요.


 ● Fine-tuning 전략을 선택해주세요. 8B 모델 + 단일 H100 80GB 환경에서 full FT는 8-bit optimizer + gradient checkpointing
   등 강한 메모리 절약을 적용해도 OOM 위험이 큽니다.
   → Full Fine-Tuning
 ● Qwen3-8B-Base는 chat_template이 없는 base 모델입니다. ShareGPT 포맷의 학습 데이터를 어떤 템플릿으로 변환할까요?
   → Qwen3 ChatML (Recommended)
 ● Loss 마스킹 전략과 max sequence length를 선택해주세요. OpenThoughts3는 reasoning trace가 길어 잘림이 발생할 수
   있습니다.
   → Assistant-only loss + 8192 (Recommended)
 ● 학습 모니터링과 체크포인트 저장 전략은 어떻게 할까요?
   → TensorBoard + 매 epoch 저장 (Recommended)


  ▎ 1-(A) truncation 8192 / 2-(A) paged_adamw_8bit + grad ckpt / 3 chat template + completion-only / 4 bs=1, ga=8 / 5
  ▎ warmup 0.03, wd 0.0 / 6 epoch마다 저장 (last 1), tensorboard, packing off
