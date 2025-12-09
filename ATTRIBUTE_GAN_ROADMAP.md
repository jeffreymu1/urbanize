# Attribute-Controllable GAN Implementation Roadmap

## Project Overview
Build an Attribute-Controllable GAN that can generate urban images with controllable attributes (wealthy, depressing, safety, lively, boring) using sliders.

## Feasibility Assessment: **HIGHLY FEASIBLE** ✅

### Strengths of Your Dataset
- **1.2M pairwise comparisons** - Excellent data volume
- **110K unique images** - Good diversity
- **All images already preprocessed** (64×64) - Ready to use
- **5 attributes** with balanced data:
  - safety: 364K comparisons
  - lively: 264K comparisons  
  - wealthy: 150K comparisons (your target)
  - depressing: 131K comparisons
  - boring: 126K comparisons

### Challenges & Solutions
1. ❌ **Challenge**: Pairwise comparisons don't give absolute scores
   ✅ **Solution**: Use Bradley-Terry model or ELO ratings to convert to continuous scores

2. ❌ **Challenge**: Images have multiple conflicting labels (e.g., wealthy in one comparison, not wealthy in another)
   ✅ **Solution**: Aggregate scores per image across all comparisons

3. ❌ **Challenge**: You have a trained unconditional GAN, need conditional one
   ✅ **Solution**: Build new conditional architecture (can't reuse unconditional weights effectively)

## Implementation Roadmap

### Phase 1: Data Preparation (2-3 hours)
**Goal**: Convert pairwise comparisons → attribute scores for each image

#### Step 1.1: Compute Attribute Scores
- Use Bradley-Terry model or ELO to get continuous scores (0-1) per image per attribute
- Start with "wealthy" attribute only
- Output: `data/attribute_scores_wealthy.csv` with columns: `image_id, wealthy_score`

#### Step 1.2: Create Training Dataset
- Match image files with scores
- Filter to images that have wealthy comparisons
- Split: 80% train, 10% val, 10% test

**Deliverable**: `src/compute_attribute_scores.py`

---

### Phase 2: Conditional GAN Architecture (3-4 hours)
**Goal**: Build conditional DCGAN that takes attribute as input

#### Step 2.1: Conditional Generator
Modify generator to accept: `[latent_noise (128-d), attribute_score (1-d)]`
- Concatenate → Dense layer → Rest of DCGAN architecture
- OR use Feature-wise Linear Modulation (FiLM) - more advanced

#### Step 2.2: Conditional Discriminator  
Two approaches:
- **Projection Discriminator** (recommended): Project attribute into discriminator's hidden layers
- **Simple Concatenation**: Concat attribute to image channels (less effective)

#### Step 2.3: Loss Functions
- Standard GAN loss (BCE or Wasserstein)
- Optional: Attribute consistency loss (use pre-trained attribute classifier)

**Deliverable**: `src/conditional_gan_model.py`

---

### Phase 3: Single-Attribute Training (8-12 hours training time)
**Goal**: Train on "wealthy" attribute and validate it works

#### Step 3.1: Training Script
- Modified training loop with conditional inputs
- Monitor: GAN loss + attribute consistency
- Save checkpoints every 10 epochs

#### Step 3.2: Validation
- Generate images at different wealthy scores: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- Visual inspection: Do they look progressively wealthier?
- Quantitative: Train simple wealthy classifier, check if generated images match requested scores

**Deliverable**: `src/train_conditional_gan.py`, `train_wealthy_attribute.sh`

---

### Phase 4: Interactive Interface (2-3 hours)
**Goal**: Create slider UI for image generation

#### Step 4.1: Backend Generation Script
- Load trained model
- Accept attribute value as parameter
- Generate batch of images

#### Step 4.2: Simple Web Interface
Options:
- **Gradio** (easiest, 30 min): Python library, instant UI
- **Streamlit** (easy, 1 hour): More customizable
- **Flask + HTML/JS** (3 hours): Full control

**Deliverable**: `src/interactive_demo.py`

---

### Phase 5: Multi-Attribute Extension (Optional, 1-2 weeks)
**Goal**: Control all 5 attributes simultaneously

#### Step 5.1: Re-compute Scores
- Get scores for all 5 attributes per image
- Handle missing data (some images don't have all attributes)

#### Step 5.2: Multi-Attribute Generator
- Input: `[latent (128-d), wealthy (1-d), depressing (1-d), safety (1-d), lively (1-d), boring (1-d)]`
- Total: 133-d input

#### Step 5.3: Multi-Slider Interface
- 5 sliders, one per attribute
- Real-time generation

---

## Timeline Estimate

| Phase | Time | Cumulative |
|-------|------|------------|
| Data Preparation | 2-3 hours | 3 hours |
| Architecture | 3-4 hours | 7 hours |
| Training (wealthy only) | 8-12 hours | 19 hours |
| Interface | 2-3 hours | 22 hours |
| **Total for MVP** | **~22 hours** | - |
| Multi-attribute (optional) | +1-2 weeks | - |

## Recommended Approach: Start Simple

### Week 1: Wealthy-Only MVP
1. ✅ Day 1-2: Data prep + architecture
2. ✅ Day 3: Start training overnight
3. ✅ Day 4: Evaluate results, build interface
4. ✅ Day 5: Iterate/improve if needed

### Week 2+: Expand
- Add more attributes one by one
- Improve quality/resolution
- Better UI/UX

## Key Technical Decisions

### 1. Attribute Score Computation Method
**Recommendation**: Bradley-Terry Model
- Standard method for pairwise comparison data
- Gives interpretable 0-1 scores
- Python library: `choix` or implement from scratch

### 2. Conditional Architecture
**Recommendation**: Projection Discriminator + Concatenation Generator
- Well-studied in literature (Miyato & Koyama, 2018)
- Better than simple concatenation
- Not too complex to implement

### 3. Framework
**Recommendation**: Stick with TensorFlow/Keras
- You already have working DCGAN code
- Easy to modify for conditional version

### 4. Interface
**Recommendation**: Gradio
- 10 lines of code for slider → image
- Perfect for demos/prototyping
- Can upgrade to Streamlit later

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] Can generate images conditioned on wealthy score
- [ ] Visually distinguishable difference between low/high wealthy
- [ ] Working slider interface

### Stretch Goals
- [ ] All 5 attributes controllable
- [ ] Quantitative validation (trained classifier agrees with requested attributes)
- [ ] Higher resolution (128×128 or 256×256)
- [ ] Interpolation videos showing smooth transitions

## Risk Mitigation

### Risk 1: Attributes might be subjective/noisy
**Mitigation**: Use only comparisons with strong agreement, filter "equal" results

### Risk 2: GAN training instability
**Mitigation**: Use proven architectures (DCGAN, Spectral Norm), careful hyperparameter tuning

### Risk 3: Attributes might be correlated (e.g., wealthy ↔ safe)
**Mitigation**: Start with one attribute, analyze correlations before multi-attribute

## Next Steps

Run this to get started immediately:
```bash
cd /Users/mwinter/School/School/csci1470-course/urbanize
python src/compute_attribute_scores.py --attribute wealthy --output data/wealthy_scores.csv
```

This will create the foundation for your conditional GAN!

