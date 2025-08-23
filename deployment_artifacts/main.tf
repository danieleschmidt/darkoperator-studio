
# DarkOperator Production Infrastructure
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# EKS Cluster for DarkOperator
resource "aws_eks_cluster" "darkoperator_cluster" {
  name     = "darkoperator-production"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.27"

  vpc_config {
    subnet_ids = aws_subnet.private[*].id
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
  ]
}

# Auto Scaling Group
resource "aws_autoscaling_group" "darkoperator_asg" {
  name                = "darkoperator-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns   = [aws_lb_target_group.darkoperator.arn]
  health_check_type   = "ELB"
  min_size            = 2
  max_size            = 100
  desired_capacity    = 3

  tag {
    key                 = "Name"
    value               = "darkoperator-instance"
    propagate_at_launch = true
  }
}
