provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "fraud-delay-retraining"
      ManagedBy = "terraform"
    }
  }
}
