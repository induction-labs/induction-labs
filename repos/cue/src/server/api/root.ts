import { postRouter } from "~/server/api/routers/post";
import { kueueRouter } from "~/server/api/routers/kueue";
import { trajectoryRouter } from "~/server/api/routers/trajectory";
import { clicksRouter } from "~/server/api/routers/clicks";
import { createCallerFactory, createTRPCRouter } from "~/server/api/trpc";

/**
 * This is the primary router for your server.
 *
 * All routers added in /api/routers should be manually added here.
 */
export const appRouter = createTRPCRouter({
  post: postRouter,
  kueue: kueueRouter,
  trajectory: trajectoryRouter,
  clicks: clicksRouter,
});

// export type definition of API
export type AppRouter = typeof appRouter;

/**
 * Create a server-side caller for the tRPC API.
 * @example
 * const trpc = createCaller(createContext);
 * const res = await trpc.post.all();
 *       ^? Post[]
 */
export const createCaller = createCallerFactory(appRouter);
