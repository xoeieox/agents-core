import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from docker_health import is_container_broken, has_unexpected_bind_ip, compose_recreate_command, configured_bind_ips

KNOWN_GOOD_IPS = {"203.0.113.10", "192.168.50.69", "127.0.0.1"}

RAG_PROXY_BROKEN_RUNNING_ZERO_PORTS = {
    "State": {"Running": True},
    "HostConfig": {
        "RestartPolicy": {"Name": "unless-stopped"},
        "PortBindings": {
            "8100/tcp": [{"HostIp": "203.0.113.10", "HostPort": "8100"},
                         {"HostIp": "127.0.0.1", "HostPort": "8100"}],
            "8200/tcp": [{"HostIp": "203.0.113.10", "HostPort": "8200"},
                         {"HostIp": "127.0.0.1", "HostPort": "8200"}],
            "8201/tcp": [{"HostIp": "203.0.113.10", "HostPort": "8201"},
                         {"HostIp": "127.0.0.1", "HostPort": "8201"}],
            "8202/tcp": [{"HostIp": "203.0.113.10", "HostPort": "8202"},
                         {"HostIp": "127.0.0.1", "HostPort": "8202"}],
            "8204/tcp": [{"HostIp": "203.0.113.10", "HostPort": "8204"},
                         {"HostIp": "127.0.0.1", "HostPort": "8204"}],
        },
    },
    "NetworkSettings": {"Ports": {}},  # empty despite PortBindings being configured - the bug
}

FORGEJO_BROKEN_EXITED = {
    "State": {"Running": False},
    "HostConfig": {
        "RestartPolicy": {"Name": "unless-stopped"},
        "PortBindings": {
            "22/tcp": [{"HostIp": "203.0.113.10", "HostPort": "2222"}],
            "3000/tcp": [{"HostIp": "203.0.113.10", "HostPort": "3000"}],
        },
    },
    "NetworkSettings": {"Ports": {}},  # Docker always reports empty Ports when not Running
    "Config": {"Labels": {
        "com.docker.compose.project.config_files": "/srv/fast/docker/forgejo/docker-compose.yml",
        "com.docker.compose.project.working_dir": "/srv/fast/docker/forgejo",
        "com.docker.compose.service": "forgejo",
    }},
}

FORGEJO_HEALTHY = {
    "State": {"Running": True},
    "HostConfig": {
        "RestartPolicy": {"Name": "unless-stopped"},
        "PortBindings": {
            "22/tcp": [{"HostIp": "203.0.113.10", "HostPort": "2222"}],
            "3000/tcp": [{"HostIp": "203.0.113.10", "HostPort": "3000"}],
        },
    },
    "NetworkSettings": {"Ports": {
        "22/tcp": [{"HostIp": "203.0.113.10", "HostPort": "2222"}],
        "3000/tcp": [{"HostIp": "203.0.113.10", "HostPort": "3000"}],
    }},
    "Config": {"Labels": {
        "com.docker.compose.project.config_files": "/srv/fast/docker/forgejo/docker-compose.yml",
        "com.docker.compose.project.working_dir": "/srv/fast/docker/forgejo",
        "com.docker.compose.service": "forgejo",
    }},
}

VAULT_RAG_NO_PUBLISHED_PORTS = {
    "State": {"Running": True},
    "HostConfig": {"RestartPolicy": {"Name": "unless-stopped"}, "PortBindings": {}},
    "NetworkSettings": {"Ports": {"8200/tcp": None}},  # internal-only, fronted by rag-proxy
}

VAULT_RAG_INERTIA_ON_DEMAND_STOPPED = {
    "State": {"Running": False},
    "HostConfig": {"RestartPolicy": {"Name": "no"}, "PortBindings": {}},
    "NetworkSettings": {"Ports": {}},
}

ADGUARD_HEALTHY_WITH_EXTRA_UNPUBLISHED_PORTS = {
    "State": {"Running": True},
    "HostConfig": {
        "RestartPolicy": {"Name": "unless-stopped"},
        "PortBindings": {
            "53/tcp": [{"HostIp": "192.168.50.69", "HostPort": "53"}],
            "53/udp": [{"HostIp": "192.168.50.69", "HostPort": "53"}],
            "80/tcp": [{"HostIp": "192.168.50.69", "HostPort": "3080"}],
        },
    },
    "NetworkSettings": {"Ports": {
        "3000/tcp": None, "3000/udp": None, "443/tcp": None, "443/udp": None,
        "53/tcp": [{"HostIp": "192.168.50.69", "HostPort": "53"}],
        "53/udp": [{"HostIp": "192.168.50.69", "HostPort": "53"}],
        "5443/tcp": None, "5443/udp": None, "6060/tcp": None,
        "67/udp": None, "68/udp": None,
        "80/tcp": [{"HostIp": "192.168.50.69", "HostPort": "3080"}],
        "853/tcp": None, "853/udp": None,
    }},
}

RESTART_ALWAYS_BROKEN_EXITED = {
    # Synthetic (no real RestartPolicy:"always" container was observed on BRIX 2026-07-18) -
    # proves the RestartPolicy check treats "always" the same as "unless-stopped" (Sonnet
    # spec-review flag: Docker's "always" policy also means "should be running").
    "State": {"Running": False},
    "HostConfig": {
        "RestartPolicy": {"Name": "always"},
        "PortBindings": {"9999/tcp": [{"HostIp": "203.0.113.10", "HostPort": "9999"}]},
    },
    "NetworkSettings": {"Ports": {}},
}


def test_rag_proxy_running_zero_ports_is_broken():
    assert is_container_broken(RAG_PROXY_BROKEN_RUNNING_ZERO_PORTS) is True


def test_forgejo_exited_is_broken():
    assert is_container_broken(FORGEJO_BROKEN_EXITED) is True


def test_forgejo_healthy_is_not_broken():
    assert is_container_broken(FORGEJO_HEALTHY) is False


def test_vault_rag_no_published_ports_is_not_broken():
    assert is_container_broken(VAULT_RAG_NO_PUBLISHED_PORTS) is False


def test_vault_rag_on_demand_stopped_is_not_broken():
    assert is_container_broken(VAULT_RAG_INERTIA_ON_DEMAND_STOPPED) is False


def test_adguard_extra_unpublished_ports_is_not_broken():
    assert is_container_broken(ADGUARD_HEALTHY_WITH_EXTRA_UNPUBLISHED_PORTS) is False


def test_restart_always_exited_is_broken():
    assert is_container_broken(RESTART_ALWAYS_BROKEN_EXITED) is True


def test_configured_bind_ips_mixed_tailscale_and_loopback():
    assert configured_bind_ips(RAG_PROXY_BROKEN_RUNNING_ZERO_PORTS) == {"203.0.113.10", "127.0.0.1"}


def test_has_unexpected_bind_ip_false_for_known_good():
    assert has_unexpected_bind_ip(RAG_PROXY_BROKEN_RUNNING_ZERO_PORTS, KNOWN_GOOD_IPS) is False
    assert has_unexpected_bind_ip(FORGEJO_BROKEN_EXITED, KNOWN_GOOD_IPS) is False
    assert has_unexpected_bind_ip(FORGEJO_HEALTHY, KNOWN_GOOD_IPS) is False


def test_has_unexpected_bind_ip_true_for_drifted_ip():
    drifted = {
        **FORGEJO_BROKEN_EXITED,
        "HostConfig": {
            "RestartPolicy": {"Name": "unless-stopped"},
            "PortBindings": {
                "22/tcp": [{"HostIp": "203.0.113.99", "HostPort": "2222"}],
                "3000/tcp": [{"HostIp": "203.0.113.99", "HostPort": "3000"}],
            },
        },
    }
    assert has_unexpected_bind_ip(drifted, KNOWN_GOOD_IPS) is True


def test_compose_recreate_command_for_forgejo():
    assert compose_recreate_command(FORGEJO_HEALTHY) == [
        "docker", "compose",
        "--project-directory", "/srv/fast/docker/forgejo",
        "-f", "/srv/fast/docker/forgejo/docker-compose.yml",
        "up", "-d", "--force-recreate", "forgejo",
    ]


def test_compose_recreate_command_none_without_labels():
    assert compose_recreate_command(RAG_PROXY_BROKEN_RUNNING_ZERO_PORTS) is None
