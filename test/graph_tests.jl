module GraphTests

using Test
import Juliagebra as SUT

const NH = SUT.NodeHandle
const NO_FLAGS = zero(SUT.NodeFlag)
const RENDERERS = Dict{DataType,SUT.Renderer}()
const MOD = 1_000_003

# ? ---------------------------------
# ! Test graphs
# ? ---------------------------------

const TEST_GRAPHS = [
    (name = "single",          parents = [Int[]]),
    (name = "chain",           parents = [Int[], [1], [2], [3]]),
    (name = "diamond",         parents = [Int[], [1], [1], [3, 2]]),
    (name = "tree",            parents = [Int[], [1], [1], [2], [2], [3], [3]]),
    (name = "two_roots",       parents = [Int[], Int[], [1, 2], [3], [1], [4, 5]]),
    (name = "fan_in",          parents = [Int[], Int[], Int[], Int[], [1, 2, 3, 4]]),
    (name = "fan_out",         parents = [Int[], [1], [1], [1], [1]]),
    (name = "skip_levels",     parents = [Int[], [1], [1, 2], [1, 2, 3]]),
    (name = "repeated_parent", parents = [Int[], [1, 1], [2]]),
    (name = "disconnected",    parents = [Int[], [1], Int[], [3], [4]]),
]

# Larger graph for the concurrency tests: 4 roots, then every node depends on
# its predecessor, on node i÷2 and on node i-7.
function big_graph(n::Int)
    parents = Vector{Vector{Int}}()
    for i in 1:n
        if i <= 4
            push!(parents, Int[])
        else
            push!(parents, unique([i - 1, i ÷ 2, max(1, i - 7)]))
        end
    end
    return parents
end

# ? ---------------------------------
# ! Helpers
# ? ---------------------------------

combine(args) = mod(1 + reduce(+, args; init = 0), MOD)
root_value(i::Int) = 10i
is_root(parents, i) = isempty(parents[i])

function make_callback(calls::Vector{Threads.Atomic{Int}}, i::Int)
    return (args...) -> begin
        Threads.atomic_add!(calls[i], 1)
        combine(args)
    end
end

function expected_values(parents, root_values::Vector{Int})
    vals = zeros(Int, length(parents))
    for i in eachindex(parents)
        vals[i] = is_root(parents, i) ? root_values[i] : combine(vals[parents[i]])
    end
    return vals
end
expected_values(parents) = expected_values(parents, [root_value(i) for i in eachindex(parents)])

function expected_children(parents)
    children = [Int[] for _ in parents]
    for j in eachindex(parents), p in parents[j]
        push!(children[p], j)
    end
    return children
end

# Every node reachable from i through child edges (i itself excluded).
function descendants(parents, i::Int)
    out = Set{Int}()
    for j in (i + 1):length(parents)
        if any(p -> p == i || p in out, parents[j])
            push!(out, j)
        end
    end
    return out
end

state(graph::SUT.GeometryPlotGraph, i) = (node = graph.nodes[i]; @atomic node.state)
has_render(graph::SUT.GeometryPlotGraph, i) = SUT.has_geom_flag(graph.nodes[i], SUT.NODE_UPDATE_RENDER)
calls_of(calls) = [c[] for c in calls]
reset!(calls) = foreach(c -> c[] = 0, calls)

function build_graph(parents; flags = nothing, names = nothing)
    graph = SUT.GeometryPlotGraph()
    calls = [Threads.Atomic{Int}(0) for _ in parents]
    handles = NH[]
    for i in eachindex(parents)
        root = is_root(parents, i)
        element = root ? root_value(i) : 0
        parent_h = root ? nothing : NH[handles[p] for p in parents[i]]
        callback = root ? nothing : make_callback(calls, i)
        flag = flags === nothing ? NO_FLAGS : flags[i]
        name = names === nothing ? nothing : names[i]
        push!(handles, SUT.add!(graph, element, nothing, parent_h, callback, flag, name))
    end
    return graph, handles, calls
end

function build_valid_graph(parents; kwargs...)
    graph, handles, calls = build_graph(parents; kwargs...)
    SUT.validate!(graph, handles[1], true)
    SUT.render!(graph, RENDERERS)
    reset!(calls)
    return graph, handles, calls
end

function wait_all(tasks; timeout = 60.0)
    return timedwait(() -> all(istaskdone, tasks), timeout; pollint = 0.01)
end

# ? ---------------------------------
# ! Custom element types (methods must be defined at top level)
# ? ---------------------------------

struct RenderProbe end
SUT.render_node(::RenderProbe, data::Any, ::Dict{DataType,SUT.Renderer}, id::UInt32) =
    (id = id, count = data === nothing ? 1 : data.count + 1)

struct Ticker
    value::Int
    changes::Bool
end
SUT.update(t::Ticker, delta_time::Float64) = (Ticker(t.value + t.changes, t.changes), t.changes)

# ? ---------------------------------
# ! Tests
# ? ---------------------------------

@testset verbose = true "Graph Tests" begin

    @testset "node flags" begin
        @test count_ones(SUT.NODE_UPDATE_RENDER) == 1
        @test count_ones(SUT.NODE_EVAL_ON_MAIN) == 1
        @test SUT.NODE_UPDATE_RENDER != SUT.NODE_EVAL_ON_MAIN

        node = SUT.GeometryPlotNode(nothing, nothing, NO_FLAGS)
        both = SUT.NODE_UPDATE_RENDER | SUT.NODE_EVAL_ON_MAIN
        @test !SUT.has_geom_flag(node, SUT.NODE_UPDATE_RENDER)

        @test SUT.set_geom_flags!(node, SUT.NODE_UPDATE_RENDER) == SUT.NODE_UPDATE_RENDER
        @test SUT.has_geom_flag(node, SUT.NODE_UPDATE_RENDER)
        @test !SUT.has_geom_flag(node, both)

        @test SUT.set_geom_flags!(node, SUT.NODE_EVAL_ON_MAIN) == both
        @test SUT.has_geom_flag(node, both)

        @test SUT.unset_geom_flags!(node, SUT.NODE_UPDATE_RENDER) == SUT.NODE_EVAL_ON_MAIN
        @test !SUT.has_geom_flag(node, SUT.NODE_UPDATE_RENDER)
        @test SUT.has_geom_flag(node, SUT.NODE_EVAL_ON_MAIN)

        @test SUT.unset_geom_flags!(node, SUT.NODE_UPDATE_RENDER) == SUT.NODE_EVAL_ON_MAIN # idempotent
        @test SUT.unset_geom_flags!(node, SUT.NODE_EVAL_ON_MAIN) == NO_FLAGS
    end

    @testset "node construction" begin
        f = () -> 1
        @test !SUT.is_dependent(nothing, nothing)
        @test SUT.is_dependent(f, nothing)
        @test SUT.is_dependent(nothing, NH[NH(1)])
        @test SUT.is_dependent(f, NH[NH(1)])

        independent = SUT.GeometryPlotNode(nothing, nothing, SUT.NODE_EVAL_ON_MAIN)
        @test (@atomic independent.state) == SUT.NODE_VALID
        @test independent.flags == SUT.NODE_EVAL_ON_MAIN
        @test independent.child_h === nothing
        @test !SUT.is_dependent(independent)

        dependent = SUT.GeometryPlotNode(f, NH[NH(1)], NO_FLAGS)
        @test (@atomic dependent.state) == SUT.NODE_INVALID
        @test dependent.parent_h == NH[NH(1)]
        @test SUT.is_dependent(dependent)
    end

    @testset verbose = true "add!" begin
        @testset "$(g.name)" for g in TEST_GRAPHS
            parents = g.parents
            n = length(parents)
            graph, handles, calls = build_graph(parents)
            children = expected_children(parents)
            roots = count(i -> is_root(parents, i), 1:n)

            @test [h.value for h in handles] == UInt32.(1:n)
            @test length(graph.elements) == n
            @test length(graph.render_data) == n
            @test length(graph.nodes) == n
            @test length(graph.element_labels) == n
            @test graph.invalid_count[] == n - roots
            @test graph.needs_render_count[] == roots
            @test all(c -> c == 0, calls_of(calls))   # add! never evaluates

            for i in 1:n
                node = graph.nodes[i]
                if is_root(parents, i)
                    @test node.parent_h === nothing
                    @test state(graph, i) == SUT.NODE_VALID
                    @test has_render(graph, i)
                    @test graph.elements[i] == root_value(i)
                else
                    @test node.parent_h == NH.(parents[i])
                    @test state(graph, i) == SUT.NODE_INVALID
                    @test !has_render(graph, i)
                end
                if isempty(children[i])
                    @test node.child_h === nothing
                else
                    @test node.child_h == NH.(children[i])
                end
                @test graph.render_data[i] === nothing
            end
        end

        @testset "callback without parents is dependent" begin
            graph = SUT.GeometryPlotGraph()
            h = SUT.add!(graph, 0, nothing, nothing, () -> 42, NO_FLAGS)
            @test state(graph, h) == SUT.NODE_INVALID
            @test graph.invalid_count[] == 1
            @test graph.needs_render_count[] == 0
            SUT.validate!(graph, h, true)
            @test graph.elements[h] == 42
            @test state(graph, h) == SUT.NODE_VALID
        end

        @testset "flags are stored" begin
            graph = SUT.GeometryPlotGraph()
            h = SUT.add!(graph, 1, nothing, nothing, nothing, SUT.NODE_EVAL_ON_MAIN)
            @test SUT.has_geom_flag(graph.nodes[h], SUT.NODE_EVAL_ON_MAIN)
            @test SUT.has_geom_flag(graph.nodes[h], SUT.NODE_UPDATE_RENDER)
        end

        @testset "render data is stored" begin
            graph = SUT.GeometryPlotGraph()
            h = SUT.add!(graph, 1, "data", nothing, nothing, NO_FLAGS)
            @test graph.render_data[h] == "data"
        end
    end

    @testset "element labels" begin
        int_name = SUT.edit_node_type_string(0)   # "Scalar" for Real elements

        graph, _, _ = build_graph([Int[], Int[], [1]])
        @test graph.element_labels == ["$(int_name)1", "$(int_name)2", "$(int_name)3"]

        names = ["Point", "Point", "Curve", nothing, "Point"]
        graph, _, _ = build_graph([Int[], Int[], [1, 2], [3], [4]]; names = names)
        @test graph.element_labels == ["Point1", "Point2", "Curve1", "$(int_name)1", "Point3"]
        @test graph.label_count["Point"] == 3
        @test graph.label_count["Curve"] == 1
        @test graph.label_count[int_name] == 1

        graph = SUT.GeometryPlotGraph()
        @test SUT.calc_element_label(graph, "A") == "A1"
        @test SUT.calc_element_label(graph, "A") == "A2"
        @test SUT.calc_element_label(graph, "B") == "B1"
        @test SUT.calc_element_label(graph, "A") == "A3"
    end

    @testset verbose = true "validate!" begin
        @testset "$(g.name)" for g in TEST_GRAPHS
            parents = g.parents
            n = length(parents)
            graph, handles, calls = build_graph(parents)

            for i in 1:n
                @test SUT._ready(graph, graph.nodes[i]) ==
                      (is_root(parents, i) || all(p -> is_root(parents, p), parents[i]))
            end

            SUT.validate!(graph, handles[1], true)

            @test all(i -> state(graph, i) == SUT.NODE_VALID, 1:n)
            @test all(i -> SUT._ready(graph, graph.nodes[i]), 1:n)
            @test graph.invalid_count[] == 0
            @test graph.needs_render_count[] == n
            @test all(i -> has_render(graph, i), 1:n)
            @test graph.elements == expected_values(parents)
            @test calls_of(calls) == [is_root(parents, i) ? 0 : 1 for i in 1:n]

            # Nothing is invalid -> second call is a no-op
            SUT.validate!(graph, handles[1], true)
            @test calls_of(calls) == [is_root(parents, i) ? 0 : 1 for i in 1:n]
        end

        @testset "start handle skips earlier nodes" begin
            parents = [Int[], [1], [1]]
            graph, handles, calls = build_graph(parents)
            SUT.validate!(graph, handles[3], true)
            @test state(graph, 2) == SUT.NODE_INVALID
            @test state(graph, 3) == SUT.NODE_VALID
            @test calls_of(calls) == [0, 0, 1]
            @test graph.invalid_count[] == 1
            @test graph.elements[3] == expected_values(parents)[3]

            SUT.validate!(graph, handles[1], true)
            @test state(graph, 2) == SUT.NODE_VALID
            @test calls_of(calls) == [0, 1, 1]
            @test graph.invalid_count[] == 0
        end

        @testset "incremental adds" begin
            parents = [Int[], [1], [2], [1, 3]]
            graph = SUT.GeometryPlotGraph()
            calls = [Threads.Atomic{Int}(0) for _ in parents]
            handles = NH[]
            push!(handles, SUT.add!(graph, root_value(1), nothing, nothing, nothing, NO_FLAGS))
            for i in 2:length(parents)
                h = SUT.add!(graph, 0, nothing, NH[handles[p] for p in parents[i]], make_callback(calls, i), NO_FLAGS)
                push!(handles, h)
                SUT.validate!(graph, h, true)
                @test state(graph, i) == SUT.NODE_VALID
                @test graph.invalid_count[] == 0
            end
            @test graph.elements == expected_values(parents)
            @test calls_of(calls) == [0, 1, 1, 1]
        end

        @testset "NODE_EVAL_ON_MAIN nodes wait for the main thread" begin
            parents = [Int[], [1], [1], [2, 3]]
            flags = [NO_FLAGS, NO_FLAGS, NO_FLAGS, SUT.NODE_EVAL_ON_MAIN]
            graph, handles, calls = build_graph(parents; flags = flags)

            SUT.validate!(graph, handles[1], false)
            @test state(graph, 2) == SUT.NODE_VALID
            @test state(graph, 3) == SUT.NODE_VALID
            @test state(graph, 4) == SUT.NODE_INVALID
            @test calls_of(calls) == [0, 1, 1, 0]
            @test graph.invalid_count[] == 1

            SUT.validate!(graph, handles[1], true)
            @test state(graph, 4) == SUT.NODE_VALID
            @test calls_of(calls) == [0, 1, 1, 1]
            @test graph.invalid_count[] == 0
            @test graph.elements == expected_values(parents)
        end
    end

    @testset verbose = true "invalidate!" begin
        @testset "$(g.name)" for g in TEST_GRAPHS
            parents = g.parents
            n = length(parents)
            @testset "node $i" for i in 1:n
                graph, handles, calls = build_valid_graph(parents)
                @test graph.needs_render_count[] == 0

                root_values = [root_value(j) for j in 1:n]
                if is_root(parents, i)
                    root_values[i] += 5
                    graph.elements[i] = root_values[i]
                end

                SUT.invalidate!(graph, handles[i])

                expected_invalid = descendants(parents, i)
                is_root(parents, i) || push!(expected_invalid, i)

                for j in 1:n
                    @test state(graph, j) == (j in expected_invalid ? SUT.NODE_INVALID : SUT.NODE_VALID)
                end
                @test graph.invalid_count[] == length(expected_invalid)
                @test isempty(graph.invalidate_stack)
                if is_root(parents, i)
                    @test has_render(graph, i)
                    @test graph.needs_render_count[] == 1
                else
                    @test graph.needs_render_count[] == 0
                end

                # Invalidating again must not double count
                SUT.invalidate!(graph, handles[i])
                @test graph.invalid_count[] == length(expected_invalid)
                @test graph.needs_render_count[] == (is_root(parents, i) ? 1 : 0)

                SUT.validate!(graph, handles[1], true)
                @test all(j -> state(graph, j) == SUT.NODE_VALID, 1:n)
                @test graph.invalid_count[] == 0
                @test calls_of(calls) == [j in expected_invalid ? 1 : 0 for j in 1:n]
                @test graph.elements == expected_values(parents, root_values)
                @test all(j -> has_render(graph, j) == (j in expected_invalid || j == i), 1:n)
            end
        end
    end

    @testset "rerender!" begin
        graph, handles, _ = build_valid_graph([Int[], [1], [1]])
        @test graph.needs_render_count[] == 0

        SUT.rerender!(graph, handles[2])
        @test has_render(graph, 2)
        @test !has_render(graph, 1)
        @test !has_render(graph, 3)
        @test graph.needs_render_count[] == 1

        SUT.rerender!(graph, handles[2])   # already flagged
        @test graph.needs_render_count[] == 1

        SUT.rerender!(graph, handles[1])
        @test graph.needs_render_count[] == 2
        @test all(i -> state(graph, i) == SUT.NODE_VALID, 1:3)  # rerender never invalidates
    end

    @testset "render!" begin
        graph = SUT.GeometryPlotGraph()
        handles = [SUT.add!(graph, RenderProbe(), nothing, nothing, nothing, NO_FLAGS) for _ in 1:3]
        @test graph.needs_render_count[] == 3

        @test SUT.render!(graph, RENDERERS) == true
        @test graph.render_data == [(id = UInt32(i), count = 1) for i in 1:3]
        @test graph.needs_render_count[] == 0
        @test !any(i -> has_render(graph, i), 1:3)

        @test SUT.render!(graph, RENDERERS) == false
        @test graph.render_data == [(id = UInt32(i), count = 1) for i in 1:3]

        SUT.rerender!(graph, handles[2])
        @test SUT.render!(graph, RENDERERS) == true
        @test [d.count for d in graph.render_data] == [1, 2, 1]
        @test graph.needs_render_count[] == 0

        @test SUT.render!(SUT.GeometryPlotGraph(), RENDERERS) == false
    end

    @testset "render! after validate! renders every evaluated node" begin
        parents = TEST_GRAPHS[3].parents   # diamond
        graph, handles, _ = build_graph(parents)
        SUT.validate!(graph, handles[1], true)
        @test graph.needs_render_count[] == length(parents)
        @test SUT.render!(graph, RENDERERS) == true
        @test graph.needs_render_count[] == 0
        @test !any(i -> has_render(graph, i), eachindex(parents))
    end

    @testset "update!" begin
        graph = SUT.GeometryPlotGraph()
        calls = [Threads.Atomic{Int}(0) for _ in 1:4]
        h1 = SUT.add!(graph, Ticker(0, true), nothing, nothing, nothing, NO_FLAGS)
        h2 = SUT.add!(graph, Ticker(0, false), nothing, nothing, nothing, NO_FLAGS)
        h3 = SUT.add!(graph, 0, nothing, NH[h1], (t) -> (Threads.atomic_add!(calls[3], 1); 2 * t.value), NO_FLAGS)
        h4 = SUT.add!(graph, 0, nothing, NH[h2], (t) -> (Threads.atomic_add!(calls[4], 1); 3 * t.value), NO_FLAGS)
        SUT.validate!(graph, h1, true)
        SUT.render!(graph, RENDERERS)
        reset!(calls)

        SUT.update!(graph, 0.016, h1)
        @test graph.elements[h1] == Ticker(1, true)
        @test graph.elements[h2] == Ticker(0, false)
        @test state(graph, h3) == SUT.NODE_INVALID
        @test state(graph, h4) == SUT.NODE_VALID
        @test graph.invalid_count[] == 1
        @test has_render(graph, 1)
        @test graph.needs_render_count[] == 1

        SUT.validate!(graph, h1, true)
        @test graph.elements[h3] == 2
        @test graph.elements[h4] == 0
        @test calls_of(calls) == [0, 0, 1, 0]

        # Starting after the changing ticker leaves it untouched
        SUT.update!(graph, 0.016, h2)
        @test graph.elements[h1] == Ticker(1, true)
        @test graph.invalid_count[] == 0

        # Non-changing elements never invalidate anything
        graph2, handles2, _ = build_valid_graph(TEST_GRAPHS[4].parents)
        SUT.update!(graph2, 0.016, handles2[1])
        @test graph2.invalid_count[] == 0
        @test graph2.needs_render_count[] == 0
    end

    @testset "clear!" begin
        graph, _, _ = build_graph(TEST_GRAPHS[4].parents)
        @test SUT.clear!(graph) === nothing
        @test isempty(graph.elements)
        @test isempty(graph.render_data)
        @test isempty(graph.nodes)
        @test isempty(graph.element_labels)
    end

    @testset "WaitPool" begin
        pool = SUT.WaitPool()
        @test length(pool.conditions) == 32
        @test SUT._wait_pool_condition(pool, 1) === SUT._wait_pool_condition(pool, 33)
        @test SUT._wait_pool_condition(pool, 1) !== SUT._wait_pool_condition(pool, 2)

        valid = SUT.GeometryPlotNode(nothing, nothing, NO_FLAGS)
        @test wait(pool, valid, 1) === nothing   # returns immediately

        node = SUT.GeometryPlotNode(() -> 1, nothing, NO_FLAGS)
        waiter = Threads.@spawn wait(pool, node, 5)
        sleep(0.1)
        @test !istaskdone(waiter)
        @atomic :release node.state = SUT.NODE_VALID
        notify(pool, 5)
        @test wait_all([waiter]; timeout = 10.0) == :ok
    end

    @testset "LockRW" begin
        rw = SUT.LockRW()
        SUT.lock_read(rw)
        SUT.lock_read(rw)
        @test rw.readers == 2
        @test !rw.writer_active

        writer = Threads.@spawn (SUT.lock_write(rw); SUT.unlock_write(rw))
        sleep(0.1)
        @test !istaskdone(writer)       # blocked by readers
        SUT.unlock_read(rw)
        sleep(0.05)
        @test !istaskdone(writer)       # one reader still active
        SUT.unlock_read(rw)
        @test wait_all([writer]; timeout = 10.0) == :ok
        @test rw.readers == 0
        @test !rw.writer_active

        SUT.lock_write(rw)
        reader = Threads.@spawn (SUT.lock_read(rw); SUT.unlock_read(rw))
        sleep(0.1)
        @test !istaskdone(reader)       # blocked by writer
        SUT.unlock_write(rw)
        @test wait_all([reader]; timeout = 10.0) == :ok
        @test rw.readers == 0
    end

    @testset verbose = true "concurrent validate!" begin
        parents = big_graph(400)
        n = length(parents)
        ntasks = max(4, 2 * Threads.nthreads())

        graph, handles, calls = build_graph(parents)
        tasks = [Threads.@spawn SUT.validate!(graph, handles[1], false) for _ in 1:ntasks]
        @test wait_all(tasks) == :ok
        @test !any(istaskfailed, tasks)
        @test all(i -> state(graph, i) == SUT.NODE_VALID, 1:n)
        @test graph.invalid_count[] == 0
        @test graph.needs_render_count[] == n
        @test calls_of(calls) == [is_root(parents, i) ? 0 : 1 for i in 1:n]
        @test graph.elements == expected_values(parents)

        SUT.render!(graph, RENDERERS)
        root_values = [root_value(i) for i in 1:n]
        for round in 1:5
            reset!(calls)
            root = mod1(round, 4)
            root_values[root] += round
            graph.elements[root] = root_values[root]
            SUT.invalidate!(graph, handles[root])
            expected_invalid = descendants(parents, root)
            @test graph.invalid_count[] == length(expected_invalid)

            tasks = [Threads.@spawn SUT.validate!(graph, handles[1], false) for _ in 1:ntasks]
            @test wait_all(tasks) == :ok
            @test !any(istaskfailed, tasks)
            @test graph.invalid_count[] == 0
            @test calls_of(calls) == [i in expected_invalid ? 1 : 0 for i in 1:n]
            @test graph.elements == expected_values(parents, root_values)
            SUT.render!(graph, RENDERERS)
        end
    end
end

end # module GraphTests
