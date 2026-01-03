#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

void away_from_high_loop(
    int32_t *flowdirs, size_t num_rows, size_t num_cols,
    int32_t *flat_mask,
    int32_t *flat_height,
    int32_t *labels,
    int32_t *high_edges, size_t *num_high_edges,
    int32_t *offsets, size_t num_offsets)
{
    size_t ci, cj, cij, ni, nj, nij, di, dj, k;
    size_t max_len = num_rows * num_cols * 2;
    int32_t loops = 1;
    int32_t label_idx;

    // Append sentinel marker (-1, -1) to the high_edges queue
    high_edges[*num_high_edges * 2] = -1;
    high_edges[*num_high_edges * 2 + 1] = -1;
    (*num_high_edges)++;

    size_t head = 0; // Head of the queue
    while (head < *num_high_edges - 1)
    {
        // Pop the first element from the queue
        ci = high_edges[head * 2];
        cj = high_edges[head * 2 + 1];
        cij = ci * num_cols + cj;
        head++;

        // Check for sentinel marker
        if (ci == (size_t)-1 && cj == (size_t)-1)
        {
            if (head == *num_high_edges - 1)
            {
                break; // No more elements in the queue
            }
            loops++;
            high_edges[*num_high_edges * 2] = -1;
            high_edges[*num_high_edges * 2 + 1] = -1;
            (*num_high_edges)++;
            continue;
        }

        // Skip if already visited
        if (flat_mask[cij] != 0)
        {
            continue;
        }

        // Mark the current cell
        flat_mask[cij] = loops;
        label_idx = labels[cij];
        flat_height[label_idx] = loops;

        // Process neighbors
        for (k = 0; k < num_offsets; k++)
        {
            di = offsets[k * 2];
            dj = offsets[k * 2 + 1];

            ni = ci + di;
            nj = cj + dj;

            // Check bounds
            if (ni < 0 || ni >= num_rows || nj < 0 || nj >= num_cols)
            {
                continue;
            }

            nij = ni * num_cols + nj;

            // Check if the neighbor belongs to the same label
            if (labels[nij] != label_idx)
            {
                continue;
            }

            // Check if the neighbor already has a flow direction
            if (flowdirs[nij] != 0)
            {
                continue;
            }

            // Add the neighbor to the queue
            high_edges[*num_high_edges * 2] = ni;
            high_edges[*num_high_edges * 2 + 1] = nj;
            (*num_high_edges)++;

            // Check for infinite loop
            if (*num_high_edges > max_len)
            {
                return; // Exit early to avoid infinite loop
            }
        }
    }
}

// Wrapper function for away_from_high_loop
static PyObject *py_away_from_high_loop(PyObject *self, PyObject *args)
{
    PyArrayObject *flowdirs_obj, *flat_mask_obj, *flat_height_obj, *labels_obj, *offsets_obj;
    PyObject *high_edges_obj;

    // Parse Python arguments
    if (!PyArg_ParseTuple(
            args, "OOOOOO",
            &flowdirs_obj, &flat_mask_obj, &flat_height_obj, &labels_obj, &high_edges_obj, &offsets_obj))
    {
        return NULL;
    }

    // Ensure inputs are NumPy arrays
    if (!PyArray_Check(flowdirs_obj) || !PyArray_Check(flat_mask_obj) ||
        !PyArray_Check(flat_height_obj) || !PyArray_Check(labels_obj) || !PyArray_Check(offsets_obj))
    {
        PyErr_SetString(PyExc_TypeError, "All inputs except high_edges must be NumPy arrays.");
        return NULL;
    }

    // Get pointers to NumPy array data
    int32_t *flowdirs = (int32_t *)PyArray_DATA(flowdirs_obj);
    int32_t *flat_mask = (int32_t *)PyArray_DATA(flat_mask_obj);
    int32_t *flat_height = (int32_t *)PyArray_DATA(flat_height_obj);
    int32_t *labels = (int32_t *)PyArray_DATA(labels_obj);
    int32_t *offsets = (int32_t *)PyArray_DATA(offsets_obj);

    // Get dimensions
    npy_intp *flowdirs_shape = PyArray_DIMS(flowdirs_obj);
    size_t num_rows = flowdirs_shape[0];
    size_t num_cols = flowdirs_shape[1];
    size_t num_offsets = PyArray_DIMS(offsets_obj)[0];

    // Convert high_edges (Python list) to a C array
    Py_ssize_t num_high_edges = PyList_Size(high_edges_obj);
    int32_t *high_edges = malloc(num_high_edges * 2 * sizeof(int32_t));
    if (high_edges == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for high_edges.");
        return NULL;
    }
    for (Py_ssize_t i = 0; i < num_high_edges; i++)
    {
        PyObject *edge = PyList_GetItem(high_edges_obj, i);
        if (!PyTuple_Check(edge) || PyTuple_Size(edge) != 2)
        {
            free(high_edges);
            PyErr_SetString(PyExc_TypeError, "Each high_edge must be a tuple of two integers.");
            return NULL;
        }
        high_edges[i * 2] = (int32_t)PyLong_AsLong(PyTuple_GetItem(edge, 0));
        high_edges[i * 2 + 1] = (int32_t)PyLong_AsLong(PyTuple_GetItem(edge, 1));
    }

    // Call the C function
    away_from_high_loop(
        flowdirs, num_rows, num_cols,
        flat_mask, flat_height, labels,
        high_edges, &num_high_edges,
        offsets, num_offsets);

    // Convert high_edges back to a Python list
    PyObject *new_high_edges = PyList_New(num_high_edges);
    for (Py_ssize_t i = 0; i < num_high_edges; i++)
    {
        PyObject *edge = PyTuple_Pack(2,
                                      PyLong_FromLong(high_edges[i * 2]),
                                      PyLong_FromLong(high_edges[i * 2 + 1]));
        PyList_SetItem(new_high_edges, i, edge);
    }

    // Free allocated memory
    free(high_edges);

    // Return the updated flat_mask, flat_height, and high_edges
    return Py_BuildValue("OO", flat_mask_obj, flat_height_obj);
}

// Method definitions
static PyMethodDef AwayFromHighLoopMethods[] = {
    {"away_from_high_loop", py_away_from_high_loop, METH_VARARGS, "Run the away_from_high_loop function."},
    {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef awayfromhighloopmodule = {
    PyModuleDef_HEAD_INIT,
    "away_from_high_loop", // Module name
    NULL,                  // Module documentation
    -1,                    // Size of per-interpreter state of the module
    AwayFromHighLoopMethods};

// Module initialization
PyMODINIT_FUNC PyInit_away_from_high_loop(void)
{
    import_array(); // Initialize NumPy
    return PyModule_Create(&awayfromhighloopmodule);
}