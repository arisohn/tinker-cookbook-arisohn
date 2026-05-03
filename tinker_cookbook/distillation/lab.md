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
- max sequence length는 8192 입니다.
- precision은 bf16을 사용해주세요.
- linger kenrnel을 사용하지 않습니다.
- 유저의 결정이 필요한 사항들을 유저에게 문의하세요.



 ● Qwen3-8B-Base는 공식 chat template이 없습니다. 데이터셋의 ShareGPT 포맷(human/gpt)을 어떤 포맷으로 변환해서
   학습할까요?
   → Qwen3 ChatML 포맷 (Recommended)
 ● Loss를 어디에 계산할까요?
   → Assistant turn에만 (completion-only) (Recommended)
 ● 1×H200(143GB)에서 effective batch size를 어떻게 가져갈까요? (gradient_checkpointing은 메모리상 필수)
   → per_device=1, grad_accum=8 → effective 8 (Recommended)
 ● Sample packing(여러 짧은 샘플을 8192까지 이어붙임)을 사용할까요?
   → Packing 끄기 (Recommended)


 ● 8192 토큰을 초과하는 샘플(추론 트레이스로 긴 샘플이 상당수)은 어떻게 처리할까요?
   → 제거 (filter out) (Recommended)
 ● Liger Kernel(Qwen3 전용 삼각형 커널/RMSNorm/RoPE 수정) 사용 여부?
   → 사용 (Recommended)
 ● 체크포인트 저장 전략은?
   → 최종 체크포인트만
