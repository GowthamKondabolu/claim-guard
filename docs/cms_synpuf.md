# CMS DE-SynPUF ingestion

ClaimGuard supports the CMS Linkable 2008–2010 Medicare Data Entrepreneurs’ Synthetic Public Use File (DE-SynPUF). CMS created these files for software development, researcher training, and privacy-safe data-mining innovation. They are realistic synthetic records, not evidence about actual Medicare beneficiaries.

Official resources:

- [CMS DE-SynPUF overview and downloads](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf)
- [CMS DE-SynPUF codebook](https://www.cms.gov/files/document/de-10-codebook.pdf-0)

## Supported files

| CMS file | ClaimGuard type | Payment mapping | Provider mapping |
|---|---|---|---|
| Inpatient claims | `inpatient` | `CLM_PMT_AMT` | `PRVDR_NUM`, then physician NPI fallback |
| Outpatient claims | `outpatient` | `CLM_PMT_AMT` | `PRVDR_NUM`, then physician NPI fallback |
| Carrier claims | `carrier` | Sum of `LINE_NCH_PMT_AMT_*` | First populated performing NPI, then tax number |
| Prescription Drug Events | `pharmacy` or `pde` | `TOT_RX_CST_AMT` | Unique no-provider sentinel |

The PDE file does not contain a provider identifier. ClaimGuard deliberately uses a unique no-provider value instead of treating a product identifier as a provider. This prevents unrelated prescriptions from creating misleading provider-utilization features.

## Convert a CMS file

Download and unzip one official CMS sample. Start with an inpatient or outpatient sample because carrier and PDE files are much larger.

```bash
claimguard ingest-cms \
  --input data/cms/DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv \
  --claim-type outpatient \
  --output data/processed/cms_outpatient_claims.csv
```

The adapter preserves source IDs in prefixed form, parses `YYYYMMDD` dates, calculates code counts, collapses repeated claim segments, and validates the canonical schema.

## Train on converted claims

```bash
claimguard train-csv \
  --input data/processed/cms_outpatient_claims.csv \
  --config configs/cms.yaml
```

The CMS configuration keeps its model, metrics, raw canonical copy, and scored output separate from the generated Phase 1 baseline.

CMS DE-SynPUF does not provide confirmed anomaly or fraud labels. The resulting metrics therefore report rows and flagged claims with `evaluation_available: 0`. Precision, recall, ROC-AUC, and average precision remain available only for the injected synthetic evaluation dataset.

## Important limitations

- Do not make population-level clinical or payment conclusions from DE-SynPUF.
- An anomaly is not fraud, waste, or abuse.
- Claim identifiers are prefixed to avoid collisions across file types.
- Repeated `SEGMENT` rows are collapsed conservatively with maximum claim-level values.
- The pandas adapter is appropriate for development samples. Large multi-sample processing belongs in the planned PySpark pipeline.
