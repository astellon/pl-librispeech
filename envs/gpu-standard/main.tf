terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.20.0"
    }
  }
}

provider "google" {
  project = "computing-cluster"
  region  = "us-central1"
}

resource "google_service_account" "default" {
  account_id   = "computing-cluster"
  display_name = "Computing Cluster"
}

resource "google_compute_instance" "default" {
  name         = "default-gpu-executer"
  machine_type = "n1-standard-8"
  zone         = "us-central1-a"

  advanced_machine_features {
    threads_per_core = 1
  }

  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  scheduling {
    preemptible        = true
    automatic_restart  = false
    provisioning_model = "SPOT"
  }

  boot_disk {
    initialize_params {
      size  = 32
      type  = pd-ssd
      image = "projects/computing-cluster/global/images/gpu-conda"
    }
  }

  network_interface {
    network = "default"
  }

  service_account {
    email  = google_service_account.default.email
    scopes = ["cloud-platform"]
  }

  metadata_startup_script = file("../run.sh")
}
