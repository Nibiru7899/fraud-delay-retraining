output "artifacts_bucket_name" {
  description = "Name of the private S3 bucket for MLflow artifacts and model files."
  value       = aws_s3_bucket.artifacts.bucket
}
