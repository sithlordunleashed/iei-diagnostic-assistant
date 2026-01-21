# IEI Diagnostic Assistant 🧬

An intelligent diagnostic chatbot for Inborn Errors of Immunity (IEI) powered by Shannon's Information Theory and clinical pattern recognition.

## Features

- **Shannon Entropy-Based Questioning**: Optimizes diagnostic efficiency by maximizing information gain
- **Bayesian Probability Updates**: Real-time refinement of differential diagnosis
- **Pathognomonic Pattern Recognition**: Instantly identifies classic syndrome constellations
- **Weighted Question Selection**: Strategic routing through nodal decision points
- **Interactive Visualizations**: Real-time probability distributions and entropy tracking
- **Clinical Decision Support**: Evidence-based diagnostic reasoning

## Quick Start

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the app:**
```bash
streamlit run app.py
```

3. **Open your browser:**
- The app will automatically open at `http://localhost:8501`

### Cloud Deployment (Free!)

#### Option 1: Streamlit Community Cloud (Recommended)

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit: IEI Diagnostic Assistant"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"
   - Your app will be live at: `https://YOUR-APP-NAME.streamlit.app`

3. **Embed in webpage:**
```html
<iframe 
    src="https://YOUR-APP-NAME.streamlit.app?embedded=true" 
    width="100%" 
    height="800px" 
    frameborder="0">
</iframe>
```

#### Option 2: Hugging Face Spaces

1. **Create account** at https://huggingface.co
2. **Create new Space** (select Streamlit as SDK)
3. **Upload files:**
   - `app.py`
   - `iei_diagnostic_engine.py`
   - `requirements.txt`
4. **Space will auto-deploy** at `https://huggingface.co/spaces/YOUR-USERNAME/iei-diagnostic`

## Project Structure

```
.
├── app.py                      # Streamlit web interface
├── iei_diagnostic_engine.py    # Core diagnostic engine
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Architecture

### Core Components

1. **Diagnostic Engine** (`iei_diagnostic_engine.py`)
   - Shannon entropy calculations
   - Bayesian probability updates
   - Pathognomonic pattern matching
   - Weighted question selection
   - 8 IEI category classifications

2. **Web Interface** (`app.py`)
   - Interactive question flow
   - Real-time probability visualization
   - Case history tracking
   - Diagnostic reasoning display

### Mathematical Framework

**Information Gain:**
```
IG(Q) = H(current) - Σ P(answer) × H(posterior | answer)
```

**Bayesian Update:**
```
P(category | answer) = P(answer | category) × P(category) / P(answer)
```

**Weighted Selection:**
```
Weighted_IG = base_IG × relevance_weight × nodal_weight
```

## Current Implementation

### Questions Implemented (10/47)
- Q15: Microbe types (highest IG: 2.95 bits)
- Q1: Age at onset
- Q12: Dysgammaglobulinemia
- Q3: Recurrent infections
- Q9: Recurrent fever
- Q5: Vaccine reactions
- Q17: Cytopenias
- Q4: Infection sites
- Q10: Chronic mucocutaneous candidiasis
- Q8: History of abscesses

### Pathognomonic Patterns (7)
- Ataxia-Telangiectasia
- Wiskott-Aldrich Syndrome
- LAD-1
- Chronic Granulomatous Disease
- SCID
- APECED (APS-1)
- Hyper-IgE Syndrome (STAT3)

### IEI Categories (8)
1. Combined Immunodeficiencies
2. Antibody Deficiencies
3. Phagocyte Defects
4. Complement Deficiencies
5. Autoinflammatory Disorders
6. Immune Dysregulation
7. Innate Immunity Defects
8. Bone Marrow Failure Syndromes

## Roadmap

### Phase 1: Core Engine (✅ COMPLETE)
- [x] Shannon entropy calculations
- [x] Bayesian updates
- [x] Pattern recognition
- [x] Weighted question selection
- [x] Top 10 questions implemented

### Phase 2: Interface (✅ COMPLETE)
- [x] Streamlit web app
- [x] Interactive questioning
- [x] Real-time visualizations
- [x] Case history tracking

### Phase 3: Expansion (NEXT)
- [ ] Add remaining 37 questions
- [ ] Expand to 47 total questions
- [ ] Add more pathognomonic patterns (15-20 total)
- [ ] Refine Fermi estimations
- [ ] Add specific disease outcomes (within categories)

### Phase 4: Advanced Features
- [ ] Case export (PDF reports)
- [ ] Multi-language support
- [ ] Mobile app (PWA)
- [ ] Integration with EMR systems
- [ ] Validation against clinical cases
- [ ] Machine learning refinement

## Usage

### Starting a Case

1. Open the app
2. (Optional) Enter patient ID
3. Answer questions as they appear
4. Watch probabilities update in real-time

### Understanding Results

- **Entropy**: Measure of diagnostic uncertainty (0-3 bits)
  - High entropy = many possibilities
  - Low entropy = narrow differential

- **Probability Distribution**: Shows likelihood across IEI categories
  - Updates after each answer using Bayes' theorem

- **Pattern Detection**: Automatically identifies pathognomonic findings
  - Triggers confirmation questions when detected

### Stopping Criteria

Diagnosis reaches conclusion when:
- Maximum probability ≥ 95%, OR
- Entropy < 0.3 bits, OR
- Pathognomonic pattern detected (confidence ≥ 90%)

## Clinical Validation

**Status**: Prototype/Research Tool

**Validation Plan:**
1. Retrospective validation on confirmed IEI cases
2. Comparison with expert clinician diagnoses
3. Prospective pilot in clinical setting
4. Iterative refinement based on outcomes

**Current Use**: Research and educational purposes only

## Contributing

This is an active research project. Contributions welcome for:
- Additional question development
- Probability refinement
- Pattern identification
- Clinical validation
- User interface improvements

## Citation

If you use this tool in research, please cite:

```
[Your manuscript details here]
Applying Information Theory to Validate Clinical Diagnostic Reasoning in IEI
Journal of Allergy and Clinical Immunology (under review)
```

## License

[To be determined - likely academic/research license]

## Acknowledgments

- IUIS 2024 Classification Committee
- ESID Registry contributors
- LASID Registry Committee
- National Institute of Pediatrics, Mexico City
- Claude.ai (Anthropic) for development assistance

## Contact

**Principal Investigator:**
Dr. Saul [Last Name]
National Institute of Pediatrics, Mexico City
President, LASID Registry Committee

## Disclaimer

⚠️ **IMPORTANT**: This is a clinical decision support tool for research and educational purposes. It does NOT replace:
- Comprehensive clinical evaluation
- Laboratory confirmation
- Genetic testing
- Expert consultation
- Definitive diagnosis by qualified specialists

Always confirm suspected diagnoses with appropriate workup and specialist review.

## Technical Support

For questions or issues:
1. Check this README
2. Review code comments in `iei_diagnostic_engine.py`
3. Test locally before deploying
4. Consult Streamlit documentation: https://docs.streamlit.io

---

**Version**: 1.0.0 (Prototype)  
**Last Updated**: January 2025  
**Status**: Active Development
