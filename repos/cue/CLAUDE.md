# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Package Manager**: Use `pnpm` for all package operations (specified in package.json)

**Development Server**: 
```bash
pnpm dev          # Start Next.js dev server with Turbo
```

**Building & Quality Checks**:
```bash
pnpm build        # Production build
pnpm check        # Run linting and type checking together
pnpm typecheck    # TypeScript type checking only
pnpm lint         # ESLint checking
pnpm lint:fix     # Auto-fix ESLint issues
```

**Code Formatting**:
```bash
pnpm format:check # Check Prettier formatting
pnpm format:write # Apply Prettier formatting
```

**Database Operations** (Drizzle ORM):
```bash
pnpm db:generate  # Generate migration files
pnpm db:migrate   # Run migrations
pnpm db:push      # Push schema changes directly
pnpm db:studio    # Open Drizzle Studio
```

## Architecture Overview

This is a **T3 Stack** application with the following key technologies:
- **Next.js 15** (App Router) - React framework
- **tRPC** - Type-safe API layer
- **Drizzle ORM** - Database toolkit with PostgreSQL
- **NextAuth.js** - Authentication
- **Tailwind CSS** - Styling
- **shadcn/ui** - UI component library built on Radix UI

### Core Application Structure

**Dual Purpose Application**:
1. **Kueue Queue Viewer** - Kubernetes job queue monitoring (main landing page)
2. **Trajectory Data Explorer** - ML training data visualization and analysis

**Routing Structure**:
- `/` - Kueue jobs dashboard and table
- `/jobs/[namespace]/[name]` - Individual job details
- `/trajectories` - Trajectory file explorer and upload interface
- `/trajectories/[gsUrl]` - URL-encoded GCS path viewer with tabs:
  - `/trajectories/[gsUrl]` - Overview tab (default)
  - `/trajectories/[gsUrl]/stats` - Statistics with confidence intervals and visualizations
  - `/trajectories/[gsUrl]/attempts/[attemptId]` - Individual trajectory step viewer

### Data Integration

**Google Cloud Storage**: Primary data source for trajectory files
- JSONL files contain trajectory records with metadata
- Separate metadata files: `metadata/{attemptId}.json` (steps) and `train_samples/{attemptId}.metadata.json` (metadata)
- GCS paths are URL-encoded in route parameters

**Kubernetes Integration**: 
- Uses `@kubernetes/client-node` to connect to cluster
- Monitors Kueue job queues and statuses
- Real-time job status tracking and queue management

### Key Backend Patterns

**tRPC Routers** (`src/server/api/routers/`):
- `kueue.ts` - Kubernetes job operations
- `trajectory.ts` - GCS data fetching and processing
- `post.ts` - Standard CRUD operations

**External Service Clients** (`src/lib/`):
- `gcs.ts` - Google Cloud Storage operations
- `kubernetes.ts` - Kubernetes cluster communication

**Schema Validation**: Zod schemas in `src/lib/schemas/` ensure type safety across API boundaries

### Frontend Architecture

**Server Components**: Layout components fetch data server-side for better performance
**Client Components**: Interactive elements use tRPC React Query hooks
**Context Pattern**: Trajectory data shared via React Context in nested routes
**Statistical Analysis**: Bootstrap confidence intervals and best@k performance metrics
**Data Visualization**: Recharts for trajectory length distribution and performance charts

### UI Components

**Base Components**: Located in `src/components/ui/` - shadcn/ui components with Radix UI primitives
**Feature Components**: Business logic components in `src/components/` and per-route `_components/`
**Styling**: Tailwind CSS with custom design system, Geist font family

### Database Schema

Uses **multi-project schema** pattern with `cue_` prefix for all tables:
- Standard NextAuth.js tables (users, accounts, sessions, verification_tokens)
- Example posts table for demonstration
- All tables use Drizzle's `createTable` helper with the prefix

### Development Environment

**DevEnv**: Uses Nix-based development environment (devenv.nix/devenv.yaml)
**Database Setup**: PostgreSQL via `start-database.sh` script
**Authentication**: NextAuth.js configuration in `src/server/auth/`

## Key Implementation Notes

**URL Encoding**: GCS paths are URL-encoded when used as route parameters - always decode with `decodeURIComponent()` before processing

**Error Handling**: Trajectory metadata operations return `null` instead of throwing errors when files are missing - handle gracefully in UI

**Performance**: Large trajectory datasets use server-side processing with shared calculations via `useMemo` for statistics

**Navigation**: Keyboard shortcuts implemented using `ahooks` library (`useKeyPress`) for trajectory step navigation

**File Structure**: UI components follow shadcn/ui patterns with forwarded refs and className merging via `cn()` utility