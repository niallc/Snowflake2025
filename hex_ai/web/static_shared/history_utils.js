// Shared undo/redo helpers used by both web frontends.
(function attachHistoryUtils(globalScope) {
    'use strict';

    function identity(value) {
        return value;
    }

    function resolveGetKey(options) {
        if (options && typeof options.getKey === 'function') {
            return options.getKey;
        }
        return identity;
    }

    function dedupeConsecutive(entries, options) {
        const getKey = resolveGetKey(options);
        const deduped = [];
        let hasPrevious = false;
        let previousKey = null;
        for (const entry of entries) {
            const key = getKey(entry);
            if (!hasPrevious || key !== previousKey) {
                deduped.push(entry);
                previousKey = key;
                hasPrevious = true;
            }
        }
        return deduped;
    }

    function pushDistinct(entries, entry, options) {
        const getKey = resolveGetKey(options);
        const maxEntries = options && Number.isFinite(options.maxEntries) ? options.maxEntries : null;
        const key = getKey(entry);
        const hasTail = entries.length > 0;
        const tailKey = hasTail ? getKey(entries[entries.length - 1]) : null;
        if (!hasTail || tailKey !== key) {
            entries.push(entry);
            if (maxEntries !== null && entries.length > maxEntries) {
                entries.splice(0, entries.length - maxEntries);
            }
        }
        return entries;
    }

    function popLastDistinct(entries, currentEntry, options) {
        const getKey = resolveGetKey(options);
        const currentKey = getKey(currentEntry);
        while (entries.length > 0) {
            const candidate = entries.pop();
            if (getKey(candidate) !== currentKey) {
                return candidate;
            }
        }
        return null;
    }

    function truncateAfterLastMatch(entries, currentEntry, options) {
        const getKey = resolveGetKey(options);
        const currentKey = getKey(currentEntry);
        for (let index = entries.length - 1; index >= 0; index -= 1) {
            if (getKey(entries[index]) === currentKey) {
                entries.splice(index + 1);
                return true;
            }
        }
        return false;
    }

    globalScope.HexHistoryUtils = Object.freeze({
        dedupeConsecutive,
        pushDistinct,
        popLastDistinct,
        truncateAfterLastMatch,
    });
})(window);
