# Terraform (AWS, ap-south-1)

Local state only for now (no remote backend). Commit `.terraform.lock.hcl`; never commit `*.tfstate*` or `.terraform/`.

    terraform init       # download the AWS provider
    terraform plan       # preview changes (read-only)
    terraform apply      # create/update resources (review the plan first)

Manages: one private, versioned, AES256-encrypted S3 bucket for MLflow artifacts and model files.

The $10 AWS Budget already exists; it was created manually in the console and is NOT managed here.

Cost guardrails: no EKS, no NAT gateways, no load balancers, no always-on SageMaker endpoints.
