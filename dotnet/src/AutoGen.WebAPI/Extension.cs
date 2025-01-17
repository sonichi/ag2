﻿// Copyright (c) 2023 - 2024, Owners of https://github.com/ag2labs
// SPDX-License-Identifier: Apache-2.0
// Contributions to this project, i.e., https://github.com/ag2labs/ag2, 
// are licensed under the Apache License, Version 2.0 (Apache-2.0).
// Portions derived from  https://github.com/microsoft/autogen under the MIT License.
// SPDX-License-Identifier: MIT
// Copyright (c) Microsoft Corporation. All rights reserved.
// Extension.cs

using AutoGen.Core;
using Microsoft.AspNetCore.Builder;

namespace AutoGen.WebAPI;

public static class Extension
{
    /// <summary>
    /// Serve the agent as an OpenAI chat completion endpoint using <see cref="OpenAIChatCompletionMiddleware"/>.
    /// If the request path is /v1/chat/completions and model name is the same as the agent name,
    /// the request will be handled by the agent.
    /// otherwise, the request will be passed to the next middleware.
    /// </summary>
    /// <param name="app">application builder</param>
    /// <param name="agent"><see cref="IAgent"/></param>
    public static IApplicationBuilder UseAgentAsOpenAIChatCompletionEndpoint(this IApplicationBuilder app, IAgent agent)
    {
        var middleware = new OpenAIChatCompletionMiddleware(agent);
        return app.Use(middleware.InvokeAsync);
    }
}
