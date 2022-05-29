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
  name         = "default-cpu-executer"
  machine_type = "n2-standard-4"
  zone         = "us-central1-a"

  scheduling {
    preemptible        = true
    automatic_restart  = false
    provisioning_model = "SPOT"
  }

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
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
