﻿// Copyright (c) 2023 - 2024, Owners of https://github.com/ag2labs
// SPDX-License-Identifier: Apache-2.0
// Contributions to this project, i.e., https://github.com/ag2labs/ag2, 
// are licensed under the Apache License, Version 2.0 (Apache-2.0).
// Portions derived from  https://github.com/microsoft/autogen under the MIT License.
// SPDX-License-Identifier: MIT
// Copyright (c) Microsoft Corporation. All rights reserved.
// RoundRobinOrchestrator.cs

using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace AutoGen.Core;

/// <summary>
/// Return the next agent in a round-robin fashion.
/// <para>
/// If the last message is from one of the candidates, the next agent will be the next candidate in the list.
/// </para>
/// <para>
/// Otherwise, the first agent in <see cref="OrchestrationContext.Candidates"/> will be returned.
/// </para>
/// <para>
/// </para>
/// </summary>
public class RoundRobinOrchestrator : IOrchestrator
{
    public async Task<IAgent?> GetNextSpeakerAsync(
        OrchestrationContext context,
        CancellationToken cancellationToken = default)
    {
        var lastMessage = context.ChatHistory.LastOrDefault();

        if (lastMessage == null)
        {
            return context.Candidates.FirstOrDefault();
        }

        var candidates = context.Candidates.ToList();
        var lastAgentIndex = candidates.FindIndex(a => a.Name == lastMessage.From);
        if (lastAgentIndex == -1)
        {
            return null;
        }

        var nextAgentIndex = (lastAgentIndex + 1) % candidates.Count;
        return candidates[nextAgentIndex];
    }
}
