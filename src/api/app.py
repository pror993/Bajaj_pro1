from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import intake, clarifications

# Initialize FastAPI app
app = FastAPI(
    title="Capital ONE Agri Claims API",
    description="API for processing insurance claims with domain classification and slot extraction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(intake.router, prefix="/api", tags=["intake"])
app.include_router(clarifications.router, prefix="/api", tags=["clarifications"])

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "Capital ONE Agri Claims API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "claims-api"} 