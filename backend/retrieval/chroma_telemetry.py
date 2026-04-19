from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpProductTelemetry(ProductTelemetryClient):
    """Disable product telemetry calls in local development."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return